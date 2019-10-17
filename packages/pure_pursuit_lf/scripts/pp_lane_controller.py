#!/usr/bin/env python
import math
import time
import numpy as np
import rospy
from duckietown_msgs.msg import (BoolStamped, FSMState, Segment,
	SegmentList, Vector2D)
from duckietown_msgs.msg import Twist2DStamped, LanePose, WheelsCmdStamped, BoolStamped, FSMState, StopLineReading
from duckietown_utils.jpg import bgr_from_jpg
from sensor_msgs.msg import CompressedImage, Image
import time
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
from line_detector.line_detector_plot import color_segment, drawLines
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
# rostopic pub /default/joy_mapper_node/car_cmd

class pp_lane_controller(object):

	def __init__(self):
		self.node_name = rospy.get_name()

		self.gp_segment_list = None
		self.lane_pose = None
		self.ground_image = None
		self.min_val = 100
		self.max_val = -100
		self.lane_width = 0.4
		self.lookup_distance = 0.25
		self.v = 0.3
		self.omega_gain = 1
		# self.omega_gain = 2
		self.temp = 2

		self.dist_list = []
		self.angle_list = []
		self.commanded_v_list = []
		self.commanded_w_list = []
		self.white_buffer_list = []
		self.yellow_buffer_list = []
		self.buffer_time = 0.4 / self.v

		self.verbose = rospy.get_param('~verbose', False)

		if self.verbose:
			self.bridge = CvBridge()
			# publishers
			self.pub_path_points = rospy.Publisher("~path_points", Image, queue_size=1)

		self.not_moving = True

		# Publishers
		self.pub_car_cmd = rospy.Publisher("~car_cmd", Twist2DStamped, queue_size=1)
		
		# Subscribers
		self.sub_lane_pose = rospy.Subscriber("~lane_pose", LanePose, self.updatePose, queue_size=1)
		self.sub_filtered_lines = rospy.Subscriber("~seglist_filtered", SegmentList, self.updateFilteredLineSeg, queue_size=1)

		# safe shutdown
		rospy.on_shutdown(self.custom_shutdown)

		rospy.loginfo("[%s] Initialized " % (rospy.get_name()))

		rospy.Timer(rospy.Duration.from_sec(2.0), self.updateParams)

	def updateParams(self, _event):
		old_verbose = self.verbose
		self.verbose = rospy.get_param('~verbose', False)

		if self.verbose != old_verbose:
			
			self.bridge = CvBridge()

			self.loginfo('Verbose is now %r' % self.verbose)
			
			# publishers
			self.pub_path_points = rospy.Publisher("~path_points", Image, queue_size=1)

	def updateFilteredLineSeg(self, gp_segment_list):
		self.gp_segment_list = gp_segment_list
		
		img_size = 480
		white_points_p0 = []
		white_points_p1 = []
		yellow_points_p0 = []
		yellow_points_p1 = []
		t = time.time()
		if self.gp_segment_list is not None:
			for segment in self.gp_segment_list.segments:
				color = segment.color
				p0 = segment.points[0]
				p1 = segment.points[1]
				p0 = [p0.x, p0.y]
				p1 = [p1.x, p1.y]
				if (color == Segment.WHITE):
					white_points_p0.append(p0)
					white_points_p1.append(p1)
					self.white_buffer_list.append([p0, t])
					self.white_buffer_list.append([p1, t])
				elif (color == Segment.YELLOW):
					yellow_points_p0.append(p0)
					yellow_points_p1.append(p1)
					self.yellow_buffer_list.append([p0, t])
					self.yellow_buffer_list.append([p1, t])
				else:
					pass

		# Initializing some variables
		white_points_p0 = np.array(white_points_p0)
		white_points_p1 = np.array(white_points_p1)
		yellow_points_p0 = np.array(yellow_points_p0)
		yellow_points_p1 = np.array(yellow_points_p1)
		num_yellow = yellow_points_p0.shape[0]
		num_white = white_points_p0.shape[0]
		
		# self.loginfo('num_yellow: %s' % str(num_yellow))
		# self.loginfo('num_white: %s' % str(num_white))
		if len(self.white_buffer_list) != 0:
			self.white_buffer_list = [x for x in self.white_buffer_list if (time.time() - x[1]) <= self.buffer_time]
			white_weights = [np.exp(self.temp * (x[1] - t)) for x in self.white_buffer_list]
			white_weights = np.array(white_weights)
			white_weights /= np.sum(white_weights)
		
		if len(self.yellow_buffer_list) != 0:
			self.yellow_buffer_list = [x for x in self.yellow_buffer_list if (time.time() - x[1]) <= self.buffer_time]
			yellow_weights = [np.exp(self.temp * (x[1] - t)) for x in self.yellow_buffer_list]
			yellow_weights = np.array(yellow_weights)
			yellow_weights /= np.sum(yellow_weights)

		if self.not_moving:
			car_cmd_msg = Twist2DStamped()
			car_cmd_msg.header = gp_segment_list.header
			car_cmd_msg.v = self.v
			car_cmd_msg.omega = 0
			self.pub_car_cmd.publish(car_cmd_msg)
			self.not_moving = False
			return
		
		weighted_white = []
		weighted_yellow = []

		# fix normal direction & compute lane width
		# pairwise_dists = self._pairwiseDists(white_points_p0, yellow_point_samples)

		half_lane_width = 0.5 * self.lane_width
		
		# avg_lane_width = np.mean(lane_width)
		# if not np.isnan(avg_lane_width):
		# 	self.lane_width = 0.5 * (avg_lane_width + self.lane_width)
		
		# compute path points
		yellow_path_points = None
		white_path_points = None

		if num_white > 0:
			p0_path_points = white_points_p0 + np.array([0, half_lane_width])
			p1_path_points = white_points_p1 + np.array([0, half_lane_width])
			white_path_points = np.vstack([p0_path_points, p1_path_points])
			min_val = np.min(white_points_p0)
			max_val = np.max(white_points_p0)

		if num_yellow > 0:
			p0_path_points = yellow_points_p0 - np.array([0, half_lane_width])
			p1_path_points = yellow_points_p1 - np.array([0, half_lane_width])
			yellow_path_points = np.vstack([p0_path_points, p1_path_points])
			min_val = np.min(yellow_points_p0)
			max_val = np.max(yellow_points_p1)
		
		if yellow_path_points is not None:
			target_point = np.mean(yellow_path_points, axis=0)

		elif white_path_points is not None:
			target_point = np.mean(white_path_points, axis=0)

		else:
			return
		
		# find closest point to lookup distance
		# target_point_ind = self.findTargetPoint(path_points)


		# follow_dist = np.linalg.norm(path_points[target_point_ind])
		follow_dist = np.linalg.norm(target_point)
		# v, omega = self.pure_pursuit(path_points[target_point_ind], self.lane_pose.d, self.lane_pose.phi, follow_dist=follow_dist)
		v, omega = self.pure_pursuit(target_point, np.array([0, 0]), np.pi / 2, follow_dist=follow_dist)

		# Compute crostrack and angle error
		if self.lane_pose is not None:
			dist_error = self.lane_pose.d
			angle_error = self.lane_pose.phi
			self.dist_list.append(dist_error)
			self.angle_list.append(angle_error)
			self.commanded_w_list.append(omega)
			self.commanded_v_list.append(v)

		car_cmd_msg = Twist2DStamped()
		car_cmd_msg.header = gp_segment_list.header
		car_cmd_msg.v = v
		car_cmd_msg.omega = omega
		self.pub_car_cmd.publish(car_cmd_msg)

		if self.verbose:
			self.min_val = np.minimum(min_val, self.min_val)
			self.max_val = np.maximum(max_val, self.max_val)
			self.ground_image = np.zeros((img_size,img_size,3), np.uint8)
			min_val = self.min_val
			max_val = self.max_val

			# for point in path_points:
			# 	i, j = self.point2pixel(point, img_size, min_val, max_val)

			# 	self.ground_image[i-1:i+1, j-1:j+1, 1] = 255
			# 	# if point[0] < 0:
			# 	#     self.ground_image[i-3:i+3, j-3:j+3, 2] = 255
			# Target point
			i, j = self.point2pixel(target_point, img_size, min_val, max_val)
			self.ground_image[i-4:i+4, j-4:j+4, 2] = 255
			# Current robot point
			i, j = self.point2pixel(np.array([0, 0]), img_size, min_val, max_val)
			self.ground_image[i-4:i+4, j-4:j+4, :] = np.array([255, 255, 0])
			
			vectors = []
			if num_white > 0:
				for point in white_path_points:
					i, j = self.point2pixel(point, img_size, min_val, max_val)
					self.ground_image[i-1:i+1, j-1:j+1] = 255

			if (num_yellow > 0):
				for point in yellow_path_points:
					i, j = self.point2pixel(point, img_size, min_val, max_val)
					self.ground_image[i-1:i+1, j-1:j+1, 1:] = 255

			if self.ground_image is not None:
				image_msg_out = self.bridge.cv2_to_imgmsg(self.ground_image, "bgr8")
				self.pub_path_points.publish(image_msg_out)

	def encounteringLineHeadOn(self, white_points_p0, n_hat, t_hat):
		# closest_white_ind = np.argmin(np.linalg.norm(white_points_p0, axis=1))
		# abs_pairwise_angles = abs(t_hat[closest_white_ind].dot(t_hat.T))
		# angle_thresh = 0.4
		# count_thresh = 0.3
		# # if (np.mean(abs_pairwise_angles < 0.1) > thresh) and (np.mean(abs_pairwise_angles > 0.9) > thresh):
		# # white_points_x_avg = np.mean(white_points_p0[:, 0])
		# # white_points_x_std = np.std(white_points_p0[:, 0])
		# # white_points_spread = np.mean(abs(white_points_p0[:,0] - white_points_x_avg) > 0.2)
		# # spread_thresh = 0.4
		# return np.mean(abs_pairwise_angles < angle_thresh) > count_thresh

		white_points_norms = np.linalg.norm(white_points_p0, axis=1)
		closest_white_ind = np.argmin(white_points_norms)
		closest_point_vec = white_points_p0[closest_white_ind] / white_points_norms[closest_white_ind]
		# avg_white_points = np.mean(white_points_p0, axis=0)
		return (closest_point_vec.dot(np.array([1, 0])) > 0.9)

	def point2pixel(self, point, img_size, min_val, max_val):
		i = img_size - int((point[0] - min_val) * img_size / (max_val - min_val))
		j = img_size - int((point[1] - min_val) * img_size / (max_val - min_val))
		i = np.clip(i, 1, img_size-1)
		j = np.clip(j, 1, img_size-1)
		return (i, j)

	def _vec2angle(self, vector):
		return np.arctan2(vector[0], vector[1])

	def _pairwiseDists(self, mat1, mat2):
		"""
		mat1 -> (i, 2)
		mat2 -> (k, 2)
		=======
		dists -> (i, k)
		"""
		mat1_magnitude = np.square(mat1).sum(axis=1, keepdims=True)
		mat2_magnitude = np.square(mat2).sum(axis=1)
		TrTe = np.dot(mat1,mat2.T)
		dists = np.sqrt((mat1_magnitude + mat2_magnitude) - (2*TrTe))
		return dists

	def _mat2angles(self, mat):
		return np.arctan2(mat[:,1], mat[:,0])

	def updatePose(self, lane_pose):
		self.lane_pose = lane_pose

	def pure_pursuit(self, curve_point, pos, angle, follow_dist=0.25):
		omega = 0.
		v = self.v
		if curve_point is not None:
			path_dir = curve_point - pos
			path_dir /= np.linalg.norm(path_dir)
			alpha = angle - self._vec2angle(path_dir)
			omega = self.omega_gain * 2 * v * np.sin(alpha) / follow_dist
		return v, omega
		
	def custom_shutdown(self):
		rospy.loginfo("[%s] Shutting down..." % self.node_name)

		self.plot_crosstrack_error()
		self.plot_angle_error()
		self.plot_velocities()
		self.plot_angular_velocities()

		# Stop the duckie
		car_cmd_msg = Twist2DStamped()
		car_cmd_msg.v = 0
		car_cmd_msg.omega = 0
		self.pub_car_cmd.publish(car_cmd_msg)
		self.not_moving = True

		# Stop listening
		self.sub_lane_pose.unregister()
		self.sub_filtered_lines.unregister()

		rospy.sleep(5)    #To make sure that it gets published.
		rospy.loginfo("[%s] Shutdown" % self.node_name)
	
	def loginfo(self, s):
		rospy.loginfo('[%s] %s' % (self.node_name, s))
	
	def plot_crosstrack_error(self):
		fig = plt.figure()
		color = 'tab:blue'
		plt.plot(self.dist_list, color=color)
		controller_name = '(PP Controller)'
		fig.suptitle('crosstrack error ' + controller_name, fontsize=16)
		plt.xlabel('timestep #')
		plt.ylabel('distance (m)', color=color)
		plt.savefig('PP_crosstrack_error.jpg')
		plt.close()

	def plot_angle_error(self):
		fig = plt.figure()
		color = 'tab:green'
		plt.plot(self.angle_list, color=color)
		controller_name = '(PP Controller)'
		fig.suptitle('angle error ' + controller_name, fontsize=16)
		plt.xlabel('timestep #')
		plt.ylabel('angle (rad)', color=color)
		plt.savefig('PP_angle_error.jpg')
		plt.close()

	def plot_velocities(self):
		fig = plt.figure()
		color = 'tab:green'
		plt.plot(self.commanded_v_list, color=color)
		controller_name = '(PP Controller)'
		fig.suptitle('commanded velocities ' + controller_name, fontsize=16)
		plt.xlabel('timestep #')
		plt.ylabel('v', color=color)
		plt.savefig('PP_velocities.jpg')
		plt.close()
	
	def plot_angular_velocities(self):
		fig = plt.figure()
		color = 'tab:blue'
		plt.plot(self.commanded_w_list, color=color)
		controller_name = '(PP Controller)'
		fig.suptitle('commanded angular velocities ' + controller_name, fontsize=16)
		plt.xlabel('timestep #')
		plt.ylabel('omega', color=color)
		plt.savefig('PP_angular_velocities.jpg')
		plt.close()

class Timer():
	def __init__(self, time_amount, thresh_time=None):
		self.time_amount = time_amount
		self.start_time = None
		self.thresh_time = thresh_time

	def startTimer(self):
		if not self.running():
			self.start_time = time.time()
	
	def running(self):
		if self.start_time is None:
			return False
		return (time.time() - self.start_time) < self.time_amount
	
	def pastThreshTime(self):
		if self.thresh_time is None:
			return
		if not self.running():
			return False
		return (time.time() - self.start_time) > self.thresh_time

if __name__ == "__main__":

	rospy.init_node("pp_lane_controller", anonymous=False)

	pp_lane_controller = pp_lane_controller()
	rospy.spin()