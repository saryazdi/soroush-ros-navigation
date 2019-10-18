#!/usr/bin/env python
import math
import time
import numpy as np
import rospy
import cv2
from duckietown_msgs.msg import Twist2DStamped, LanePose, Segment, SegmentList
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os

class pp_lane_controller(object):

	def __init__(self):
		self.node_name = rospy.get_name()

		self.gp_segment_list = None
		self.ground_image = None
		self.min_val = 100
		self.max_val = -100
		self.lane_width = 0.4
		self.lookahead_distance = 0.25
		self.v = 0.44
		self.omega_gain = 3
		self.momentum = 0.8

		self.dist_list = []
		self.angle_list = []
		self.commanded_v_list = []
		self.commanded_w_list = []
		self.start_time = time.time()
		self.t_error_publish = time.time()

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
				elif (color == Segment.YELLOW):
					yellow_points_p0.append(p0)
					yellow_points_p1.append(p1)
				else:
					pass

		# Initializing some variables
		white_points_p0 = np.array(white_points_p0)
		white_points_p1 = np.array(white_points_p1)
		white_points = np.vstack([white_points_p0, white_points_p1])
		yellow_points_p0 = np.array(yellow_points_p0)
		yellow_points_p1 = np.array(yellow_points_p1)
		yellow_points = np.vstack([yellow_points_p0, yellow_points_p1])
		num_yellow = yellow_points_p0.shape[0]
		num_white = white_points_p0.shape[0]

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
		filtered_white = np.zeros(1)
		filtered_yellow = np.zeros(1)

		if (num_white > 0):
			filtered_white = white_points[np.linalg.norm(white_points, axis=1) < 0.7]
		if (num_yellow > 0):
			filtered_yellow = yellow_points[np.linalg.norm(yellow_points, axis=1) < 0.7]
		if (filtered_white.shape[0] > 2) and (filtered_yellow.shape[0] > 2):
			self.lane_width = ((1 - self.momentum) * (np.mean(filtered_yellow[:,1]) - np.mean(filtered_white[:,1]))) + (self.momentum * self.lane_width)

		half_lane_width = 0.5 * self.lane_width
		
		# compute path points
		yellow_path_points = None
		white_path_points = None

		if num_white > 0:
			if (filtered_white.shape[0] > 2):
				white_path_points = filtered_white + np.array([0, half_lane_width])
			else:
				white_path_points = white_points + np.array([0, half_lane_width])
			min_val = np.min(white_points)
			max_val = np.max(white_points)

		if num_yellow > 0:
			if (filtered_yellow.shape[0] > 2):
				yellow_path_points = filtered_yellow - np.array([0, half_lane_width])
			else:
				yellow_path_points = yellow_points - np.array([0, half_lane_width])
			min_val = np.min(yellow_points)
			max_val = np.max(yellow_points)
		
		if yellow_path_points is not None:
			target_point = np.mean(yellow_path_points, axis=0)
			trajectory_points = yellow_path_points
		elif white_path_points is not None:
			target_point = np.mean(white_path_points, axis=0)
			trajectory_points = white_path_points
		else:
			return

		heuristic0 = (1 + (np.std(trajectory_points[:,0]))) ** 4 # if we have a lot of variance in forward direction, correct less
		heuristic1 = 1. / (1 + np.std(trajectory_points[:,1])) ** 2 # if we have a lot of variance in right/left direction, correct more
		# ind_highest = np.argmax(trajectory_points[:,0])
		# future_ref = abs(trajectory_points[ind_highest, 1])
		lookahead_distance = self.lookahead_distance * heuristic0 * heuristic1
		# lookahead_distance = self.lookahead_distance / ((1 + future_ref) ** 2)
		# self.loginfo('lookahead_distance %s' % str(lookahead_distance))
		target_point = target_point * lookahead_distance / np.linalg.norm(target_point)
		v, omega = self.pure_pursuit(target_point, np.array([0, 0]), np.pi / 2, follow_dist=lookahead_distance)

		# Compute crostrack and angle error
		t = time.time()
		self.commanded_w_list.append([omega, t - self.start_time])
		self.commanded_v_list.append([v, t - self.start_time])

		if (self.verbose) and (len(self.commanded_v_list) > 3) and (len(self.dist_list) > 3):
			if ((time.time() - self.t_error_publish) > 10):
				self.saveErrors()
				self.t_error_publish = time.time()
		
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

	def updatePose(self, lane_pose):
		t = time.time()
		self.dist_list.append([lane_pose.d, t - self.start_time])
		self.angle_list.append([lane_pose.phi, t - self.start_time])

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

		self.save_crosstrack_error()
		self.save_angle_error()
		self.save_velocities()
		self.save_angular_velocities()

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
	
	def saveErrors(self):
		self.save_crosstrack_error()
		self.save_angle_error()
		self.save_velocities()
		self.save_angular_velocities()

	def imreadAndBridge(self, img_loc):
		return self.bridge.cv2_to_imgmsg(cv2.imread(img_loc), "bgr8")

	def save_crosstrack_error(self):
		if len(self.dist_list) > 150:
			ind = np.linspace(0, len(self.dist_list) - 1, num=150).astype('int32')
			points = np.array(self.dist_list)[ind]
		else:
			points = np.array(self.dist_list)
		self.saveArray(points, '/data/log/PP_crosstrack_error.txt')

	def save_angle_error(self):
		if len(self.angle_list) > 350:
			ind = np.linspace(0, len(self.angle_list) - 1, num=350).astype('int32')
			points = np.array(self.angle_list)[ind]
		else:
			points = np.array(self.angle_list)
		self.saveArray(points, '/data/log/PP_angle_error.txt')

	def save_velocities(self):
		if len(self.commanded_v_list) > 350:
			ind = np.linspace(0, len(self.commanded_v_list) - 1, num=350).astype('int32')
			points = np.array(self.commanded_v_list)[ind]
		else:
			points = np.array(self.commanded_v_list)
		self.saveArray(points, '/data/log/PP_velocities.txt')
	
	def save_angular_velocities(self):
		if len(self.commanded_w_list) > 350:
			ind = np.linspace(0, len(self.commanded_w_list) - 1, num=350).astype('int32')
			points = np.array(self.commanded_w_list)[ind]
		else:
			points = np.array(self.commanded_w_list)
		self.saveArray(points, '/data/log/PP_angular_velocities.txt')
	
	def saveArray(self, arr, filename):
		with open(filename, 'w') as f:
			for x in arr:
				f.write("%s, " % str(x[0]))
				f.write("%s\n" % str(x[1]))

if __name__ == "__main__":

	rospy.init_node("pp_lane_controller", anonymous=False)

	pp_lane_controller = pp_lane_controller()
	rospy.spin()