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

		self.segment_list = None
		self.gp_segment_list = None
		self.filtered_seg_list = None
		self.lane_pose = None
		self.ground_image = None
		self.white_line_orientation = 'RIGHT'
		self.lane_width = 0.23
		self.min_val = 100
		self.max_val = -100
		self.lane_width = 0.23
		self.lookup_distance = 0.25
		self.num_yellow_subsamples = 100
		self.num_white_subsamples = 100
		self.v = 0.3

		right_turn_time = 0.6 / self.v
		left_turn_time = 0.6 / self.v
		bad_turn_time = 1.0 / self.v

		self.encounteredRightLinetimer = Timer(right_turn_time)
		self.encounteredLeftLinetimer = Timer(left_turn_time)
		self.encounteredBadTurntimer = Timer(bad_turn_time)

		self.noiseCombatTimer = Timer(0.75)

		self.dist_list = []
		self.angle_list = []
		self.commanded_v_list = []
		self.commanded_w_list = []
		self.buffer_list = []
		self.buffer_time = 0.85 / self.v

		self.verbose = rospy.get_param('~verbose', False)

		if self.verbose:
			self.bridge = CvBridge()
			# publishers
			self.pub_path_points = rospy.Publisher("~path_points", Image, queue_size=1)
			# subscribers
			self.sub_lines = rospy.Subscriber("~segment_list", SegmentList, self.updateLineSeg, queue_size=1)

		self.not_moving = True

		# Publishers
		self.pub_car_cmd = rospy.Publisher("~car_cmd", Twist2DStamped, queue_size=1)
		
		# Subscribers
		# /default/lane_filter_node/seglist_filtered -> subscribed 2 by duckiebot_visualizer
		# /default/line_detector_node/segment_list -> subscribed 2 by ground projection
		# /default/ground_projection/lineseglist_out -> published by ground projection
		self.sub_lane_pose = rospy.Subscriber("~lane_pose", LanePose, self.updatePose, queue_size=1)
		self.sub_gp_lines = rospy.Subscriber("~lineseglist_out", SegmentList, self.updateGPLineSeg, queue_size=1)
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

			# subscribers
			self.sub_lines = rospy.Subscriber("~segment_list", SegmentList, self.updateLineSeg, queue_size=1)

	def updateLineSeg(self, segment_list):
		self.segment_list = segment_list

	def updateFilteredLineSeg(self, filtered_seg_list):
		self.filtered_seg_list = filtered_seg_list

	def updateGPLineSeg(self, gp_segment_list):
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
				else:
					pass
		
		if self.filtered_seg_list is not None:
			for segment in self.filtered_seg_list.segments:
				color = segment.color
				p0 = segment.points[0]
				p1 = segment.points[1]
				p0 = [p0.x, p0.y]
				p1 = [p1.x, p1.y]
				if (color == Segment.YELLOW):
					yellow_points_p0.append(p0)
					yellow_points_p1.append(p1)
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
		if len(self.buffer_list) != 0:
			self.buffer_list = [x for x in self.buffer_list if (time.time() - x) <= self.buffer_time]

		if self.not_moving:
			car_cmd_msg = Twist2DStamped()
			car_cmd_msg.header = gp_segment_list.header
			car_cmd_msg.v = self.v
			car_cmd_msg.omega = 0
			self.pub_car_cmd.publish(car_cmd_msg)
			self.not_moving = False
			return

		if (num_white < 2):
			return
		
		if (num_yellow < 2):
			# Compute t_hat (vector between two consecutive white points on a line) and n_hat (normal to t_hat)
			t_hat = (white_points_p1 - white_points_p0) / np.linalg.norm(white_points_p1 - white_points_p0, axis=1, keepdims=True)
			n_hat = t_hat[:, ::-1].copy()
			n_hat[:, 1] *= -1

			half_lane_width = 0.5 * self.lane_width
			if self.white_line_orientation is 'RIGHT':
				invert_ind = np.einsum('ij,ij->i', white_points_p0, n_hat) > 0
				t_hat[invert_ind] *= -1
				n_hat[invert_ind] *= -1
				normal_dir = n_hat
				left_white_line_ind = np.ones(invert_ind.shape) < 0

				# closest_white_ind = np.argmin(np.linalg.norm(white_points_p0, axis=1))
				# abs_pairwise_angles = abs(t_hat[closest_white_ind].dot(t_hat.T))
				# thresh = 0.7
				# # if (np.mean(abs_pairwise_angles < 0.1) > thresh) and (np.mean(abs_pairwise_angles > 0.9) > thresh):
				# white_points_x_avg = np.mean(white_points_p0[:, 0])
				# white_points_x_std = np.std(white_points_p0[:, 0])
				# white_points_spread = np.mean(abs(white_points_p0[:,0] - white_points_x_avg) > 0.2)
				# spread_thresh = 0.4
				# if (np.mean(abs_pairwise_angles < self.bad_turn_thresh) > thresh):
				if (self.encounteringLineHeadOn(white_points_p0, n_hat, t_hat)):
					# self.buffer_list.append(time.time())
					# self.loginfo(len(self.buffer_list))
					self.encounteredRightLinetimer.startTimer()
				
				if (self.encounteringBadTurn(white_points_p0, white_points_p1, n_hat, t_hat) or self.encounteredBadTurntimer.running()):
					self.loginfo('BADTURN!')
					self.encounteredBadTurntimer.startTimer()
					normal_dir = np.array([1, 1])


				# if len(self.buffer_list) > 2:
					# self.encounteredRightLinetimer.startTimer()
					# self.loginfo('NOPE')

				# if self.timer.running(): # beautiful open-loop hack
				# 	closest_white_ind = np.argmin(np.linalg.norm(white_points_p0, axis=1))
				# 	perp_points = abs_pairwise_angles[closest_white_ind] < self.bad_turn_thresh
				# 	n_hat[perp_points] *= -1
				# 	t_hat[perp_points] *= -1
					# half_lane_width *= 2

					# sleep_dist = 0.4
					# time.sleep(sleep_dist / self.v) 

					# corner_min = np.min(white_points_p0[perp_points + closest_white_ind], axis=0)
					# corner_max = np.max(white_points_p0[perp_points + closest_white_ind], axis=0)
					# if (corner_min[1] > 0) or (corner_max[1] < 0):
					#     n_hat[perp_points] *= -1
					#     t_hat[perp_points] *= -1
			else:
				invert_ind = np.einsum('ij,ij->i', white_points_p0, n_hat) < 0
				half_lane_width *= -3
				left_white_line_ind = np.ones(invert_ind.shape) > 0
				t_hat[invert_ind] *= -1
				n_hat[invert_ind] *= -1
				normal_dir = n_hat

				if (self.encounteringLineHeadOn(white_points_p0, n_hat, t_hat)):
					self.encounteredLeftLinetimer.startTimer()

			# compute path points
			normal_step = normal_dir * half_lane_width
			
			p0_path_points = white_points_p0 + normal_step
			p1_path_points = white_points_p1 + normal_step
			
			path_points = np.vstack([p0_path_points, p1_path_points])

			if path_points.shape[0] < 2:
				return

			target_point_ind = self.findTargetPoint(path_points)

		else:
			if self.verbose:
				# min_val and max_val are only used when visualizing the points
				max_val = np.maximum(np.max(white_points_p0), np.max(yellow_points_p0))
				min_val = np.minimum(np.min(white_points_p0), np.min(yellow_points_p0))
				self.min_val = np.minimum(min_val, self.min_val)
				self.max_val = np.maximum(max_val, self.max_val)
			
			# sample yellow and white points to make computation quicker
			if (num_yellow > self.num_yellow_subsamples):
				rand_ind = np.random.choice(num_yellow, self.num_yellow_subsamples, replace=False)
				yellow_point_samples = yellow_points_p0[rand_ind]
			else:
				yellow_point_samples = yellow_points_p0
			
			# fix normal direction & compute lane width
			pairwise_dists = self._pairwiseDists(white_points_p0, yellow_point_samples)

			# # REJECT OUTLIER YELLOW POINTS
			# # reject yellow points on opposite side of closest white point (compared to current robot position)
			# closest_white_ind = np.argmin(pairwise_dists, axis=0)
			# inlier_keep = np.einsum('ij, ij-> i', white_points_p0[closest_white_ind], white_points_p0[closest_white_ind] - yellow_point_samples) > 0
			# pairwise_dists = pairwise_dists[:, inlier_keep]
			# yellow_point_samples = yellow_point_samples[inlier_keep]

			# if pairwise_dists.shape[1] < 2:
			#     return
			
			# REJECT OUTLIER WHITE POINTS
			pairwise_dists[pairwise_dists < 0.15] = 10000 # don't choose yellow points which are too close to white points (noise)
			closest_yellow_ind = np.argmin(pairwise_dists, axis=1)
			# Sign of angles_diff helps detect right from left white lane, but if it is close to 0 it is ambiguous => reject
			angles_diff = self._mat2angles(white_points_p0) - self._mat2angles(yellow_point_samples[closest_yellow_ind])
			keep_points = abs(angles_diff) > 0.1 # ~5 degrees
			# throw away closest yellow points with large distances
			keep_points *= pairwise_dists[np.arange(closest_yellow_ind.shape[0]), closest_yellow_ind] < 10

			closest_yellow_ind = closest_yellow_ind[keep_points]
			pairwise_dists = pairwise_dists[keep_points]
			white_points_p0 = white_points_p0[keep_points]
			white_points_p1 = white_points_p1[keep_points]
			left_white_line_ind = angles_diff[keep_points] > 0
			# self.loginfo('Kept %s of %s points' % (str(np.sum(keep_points)), str(keep_points.shape[0])))

			# Compute t_hat (vector between two consecutive white points on a line) and n_hat (normal to t_hat)
			t_hat = (white_points_p1 - white_points_p0) / np.linalg.norm(white_points_p1 - white_points_p0, axis=1, keepdims=True)
			n_hat = t_hat[:, ::-1].copy()
			n_hat[:, 1] *= -1
			
			invert_ind = np.einsum('ij,ij->i', yellow_point_samples[closest_yellow_ind] - white_points_p0, n_hat) < 0
			invert_ind = np.logical_xor(invert_ind, left_white_line_ind)

			t_hat[invert_ind] *= -1
			n_hat[invert_ind] *= -1

			# compute half lane width, consider whether white line is on left side of yellow line or not
			lane_width = np.min(pairwise_dists, axis=1, keepdims=True)
			half_lane_width = 0.5 * lane_width
			half_lane_width[left_white_line_ind] *= -3
			
			avg_lane_width = np.mean(lane_width)
			if not np.isnan(avg_lane_width):
				self.lane_width = 0.5 * (avg_lane_width + self.lane_width)
			
			# what to do if lost yellow points
			if (np.sum(~left_white_line_ind) > np.sum(left_white_line_ind)):
				self.white_line_orientation = 'RIGHT'
			else:
				self.white_line_orientation = 'LEFT'
			
			# compute path points
			normal_step = n_hat * half_lane_width
			p0_path_points = white_points_p0 + normal_step
			p1_path_points = white_points_p1 + normal_step
			
			path_points = np.vstack([p0_path_points, p1_path_points])

			if path_points.shape[0] < 2:
				return
			
			# find closest point to lookup distance
			target_point_ind = self.findTargetPoint(path_points)


		follow_dist = np.linalg.norm(path_points[target_point_ind])
		# v, omega = self.pure_pursuit(path_points[target_point_ind], self.lane_pose.d, self.lane_pose.phi, follow_dist=follow_dist)
		v, omega = self.pure_pursuit(path_points[target_point_ind], np.array([0, 0]), np.pi/2, follow_dist=follow_dist)

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
			self.ground_image = np.zeros((img_size,img_size,3), np.uint8)
			min_val = self.min_val
			max_val = self.max_val

			for point in path_points:
				i, j = self.point2pixel(point, img_size, min_val, max_val)

				self.ground_image[i-1:i+1, j-1:j+1, 1] = 255
				# if point[0] < 0:
				#     self.ground_image[i-3:i+3, j-3:j+3, 2] = 255
			# Target point
			i, j = self.point2pixel(path_points[target_point_ind], img_size, min_val, max_val)
			self.ground_image[i-4:i+4, j-4:j+4, 2] = 255
			# Current robot point
			i, j = self.point2pixel(np.array([0, 0]), img_size, min_val, max_val)
			self.ground_image[i-4:i+4, j-4:j+4, :] = np.array([255, 255, 0])
			
			vectors_right = []
			vectors_left = []
			for i, line in enumerate(np.hstack([white_points_p0, white_points_p1])):
				i0, j0 = self.point2pixel(line[:2], img_size, min_val, max_val)
				i1, j1 = self.point2pixel(line[2:], img_size, min_val, max_val)
				if not invert_ind[i]:
					l = [j0, i0, j1, i1]
				else:
					l = [j1, i1, j0, i0]
				if not left_white_line_ind[i]:
					vectors_right.append(l)
				else:
					vectors_left.append(l)
			if len(vectors_right) > 0:
				drawArrowedLines(self.ground_image, np.array(vectors_right), (255,255,255), None, None)
			if len(vectors_left) > 0:
				drawArrowedLines(self.ground_image, np.array(vectors_left), (255,0,0), None, None)

			if (num_yellow > 4):
				for point in yellow_points_p0:
					i, j = self.point2pixel(point, img_size, min_val, max_val)
					self.ground_image[i-1:i+1, j-1:j+1, 1:] = 255

			if self.ground_image is not None:
				image_msg_out = self.bridge.cv2_to_imgmsg(self.ground_image, "bgr8")
				# image_msg_out.header.stamp = image_msg.header.stamp
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

	def encounteringBadTurn(self, white_points_p0, white_points_p1, n_hat, t_hat):
		p0_path_points = white_points_p0 + n_hat
		p1_path_points = white_points_p1 + n_hat
		white_points = np.vstack([white_points_p0, white_points_p1])
		path_points = np.vstack([p0_path_points, p1_path_points])
		avg_path_point = np.mean(path_points, axis=0)
		avg_white_point = np.mean(white_points, axis=0)
		return ((avg_path_point[0] < avg_white_point[0]) and (avg_path_point[1] < avg_white_point[1]))

	def point2pixel(self, point, img_size, min_val, max_val):
		i = img_size - int((point[0] - min_val) * img_size / (max_val - min_val))
		j = img_size - int((point[1] - min_val) * img_size / (max_val - min_val))
		i = np.clip(i, 1, img_size-1)
		j = np.clip(j, 1, img_size-1)
		return (i, j)

	def findTargetPoint(self, path_points):
		# find closest point to lookup distance
		path_points_copy = path_points.copy()
		# path_points_copy[path_points_copy[:,0] < 0] -= np.array([self.lookup_distance, 0])
		# path_points_copy[path_points_copy[:,0] < 0] = 100
		avg_path_point = np.mean(path_points, axis=0)
		avg_path_point_sign = np.sign(avg_path_point)
		path_points_spread = np.mean(abs(path_points[:,0] - avg_path_point[0]) > 0.2)
		spread_thresh = 0.4
		# if (path_points_spread > spread_thresh):
		if self.encounteredBadTurntimer.running():
			heuristic = 0.5 * path_points_copy[:, 1]
			target_point_ind = np.argmin(abs((path_points_copy[:, 0] ** 2) + (path_points_copy[:, 1] ** 2) - (self.lookup_distance ** 2)) + heuristic)
			self.encounteredLeftLinetimer.startTimer()
			self.loginfo('rightmost')

		elif self.encounteredRightLinetimer.running():
			# target_point_ind = np.argmax(path_points_copy[:, 1])
			heuristic = 0.5 * path_points_copy[:, 1]
			target_point_ind = np.argmin(abs((path_points_copy[:, 0] ** 2) + (path_points_copy[:, 1] ** 2) - (self.lookup_distance ** 2)) - heuristic)
			self.loginfo('leftmost')

		elif (self.encounteredLeftLinetimer.running()) or (avg_path_point_sign[0] < 0.):
			# mismatch0 = np.sign(path_points_copy[:,0]) != avg_path_point[0]
			# mismatch1 = path_points_copy[:,1] > 0
			# reject_mask = np.bitwise_or(mismatch0, mismatch1)
			# path_points_copy[reject_mask] = 100
			# target_point_ind = np.argmin(path_points_copy[:, 1])
			heuristic = 0.5 * path_points_copy[:, 1]
			target_point_ind = np.argmin(abs((path_points_copy[:, 0] ** 2) + (path_points_copy[:, 1] ** 2) - (self.lookup_distance ** 2)) + heuristic)
			self.encounteredLeftLinetimer.startTimer()
			self.loginfo('rightmost')

		# elif (abs(avg_path_point[1]) < 0.25) and ((np.max(path_points[:,0]) - np.min(path_points[:,0])) < 0.25) and (avg_path_point[1] < 0.1):
		# 	target_point_ind = np.argmin(path_points_copy[:, 1])
		# 	self.noiseCombatTimer.startTimer()
		# 	self.loginfo('Close call!')
		else:
			mismatch = np.sign(path_points_copy) != avg_path_point_sign
			reject_mask = np.bitwise_or(mismatch[:,0], mismatch[:,1])
			path_points_copy[reject_mask] = 100
			target_point_ind = np.argmin(abs((path_points_copy[:, 0] ** 2) + (path_points_copy[:, 1] ** 2) - (self.lookup_distance ** 2)))
		return target_point_ind

	def _vec2angle(self, vector):
		return np.arctan2(vector[1], vector[0])

	def _vec2angle2(self, vector):
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

	# def _rejectOutliers(self, x, m=2.):
	#     # return (abs(x - np.mean(x)) < 10000)
	#     return (abs(x - np.mean(x)) < (m * np.std(x)))

	# def _rejectOutliers2d(self, x, m=2.):
	#     return (np.linalg.norm(x - np.mean(x, axis=0), axis=1) < (m * np.std(x)))

	# def _rejectOutliers(self, x, m=5.):
	#     d = np.abs(x - np.median(x))
	#     mdev = np.median(d)
	#     s = d/mdev if mdev else 0.
	#     return x[s<m]
	
	# def _rejectOutliers(self, dataIn,factor):
	#     quant3, quant1 = np.percentile(dataIn, [75 ,25])
	#     iqr = quant3 - quant1
	#     iqrSigma = iqr/1.34896
	#     medData = np.median(dataIn)
	#     dataOut = [ x for x in dataIn if ( (x > medData - factor* iqrSigma) and (x < medData + factor* iqrSigma) ) ] 
	#     return(dataOut)

	def updatePose(self, lane_pose):
		self.lane_pose = lane_pose

	def pure_pursuit(self, curve_point, pos, angle, follow_dist=0.25):
		omega = 0.
		v = self.v
		if curve_point is not None:
			path_dir = curve_point - pos
			path_dir /= np.linalg.norm(path_dir)
			alpha = angle - self._vec2angle2(path_dir)
			omega = 2 * v * np.sin(alpha) / follow_dist
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
		self.sub_gp_lines.unregister()
		self.sub_filtered_lines.unregister()
		if self.verbose:
			self.sub_lines.unregister()

		rospy.sleep(0.5)    #To make sure that it gets published.
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

def drawArrowedLines(bgr, lines, paint, p1_color=(0,255,0), p2_color=(0,0,255)):
	if len(lines)>0:
		for x1,y1,x2,y2 in lines:
			cv2.arrowedLine(bgr, (x1,y1), (x2,y2), paint, 2)
			if p1_color is not None:
				cv2.circle(bgr, (x1,y1), 2, p1_color)
			if p2_color is not None:
				cv2.circle(bgr, (x2,y2), 2, p2_color)

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