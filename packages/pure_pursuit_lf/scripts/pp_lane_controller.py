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

class pp_lane_controller(object):

    def __init__(self):
        self.node_name = rospy.get_name()

        # self.image_size = rospy.get_param('~img_size')

        # Constructor of line detector
        self.bridge = CvBridge()

        # Publication
        self.path_trajectory = rospy.Publisher("/default/pp_lane_controller/path_trajectory", CompressedImage, queue_size=1)
        
        # Subscriptions
        self.sub_image = rospy.Subscriber("/default/anti_instagram_node/corrected_image/compressed", CompressedImage, self.processImage, queue_size=1)
        # self.sub_lines = rospy.Subscriber("/default/line_detector_node/segment_list", SegmentList, self.findTrajectory, queue_size=1)

        # safe shutdown
        rospy.on_shutdown(self.custom_shutdown)

        rospy.loginfo("[%s] Initialized " % (rospy.get_name()))

    # def findTrajectory(self, msg):
    #     self.stop_line_distance = np.sqrt(msg.stop_line_point.x**2 + msg.stop_line_point.y**2 + msg.stop_line_point.z**2)
    #     self.stop_line_detected = msg.stop_line_detected

    def processImage(self, image_msg):
        # Decode from compressed image with OpenCV
        try:
            image_cv = bgr_from_jpg(image_msg.data)
        except ValueError as e:
            self.loginfo('Could not decode image: %s' % e)
            return
        
        # Resize and crop image
        hei_original, wid_original = image_cv.shape[0:2]

        # if self.image_size[0] != hei_original or self.image_size[1] != wid_original:
        # image_cv = cv2.resize(image_cv, (self.image_size[1], self.image_size[0]),
        #                         interpolation=cv2.INTER_NEAREST)
        image_cv = cv2.resize(image_cv, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        image_cv = cv2.GaussianBlur(image_cv,(5,5),0)

         # Publish the frame with lines
        image_msg_out = self.bridge.cv2_to_imgmsg(image_cv, "bgr8")
        image_msg_out.header.stamp = image_msg.header.stamp
        self.path_trajectory.publish(image_msg_out)
        
    def custom_shutdown(self):
        rospy.loginfo("[%s] Shutting down..." % self.node_name)

        # Stop listening
        self.sub_image.unregister()

        rospy.sleep(0.5)    #To make sure that it gets published.
        rospy.loginfo("[%s] Shutdown" %self.node_name)
    
    def loginfo(self, s):
        rospy.loginfo('[%s] %s' % (self.node_name, s))

class Stats():
    def __init__(self):
        self.nresets = 0
        self.reset()

    def reset(self):
        self.nresets += 1
        self.t0 = time.time()
        self.nreceived = 0
        self.nskipped = 0
        self.nprocessed = 0

    def received(self):
        if self.nreceived == 0 and self.nresets == 1:
            rospy.loginfo('line_detector_node received first image.')
        self.nreceived += 1

    def skipped(self):
        self.nskipped += 1

    def processed(self):
        if self.nprocessed == 0 and self.nresets == 1:
            rospy.loginfo('line_detector_node processing first image.')

        self.nprocessed += 1

    def info(self):
        delta = time.time() - self.t0

        if self.nreceived:
            skipped_perc = (100.0 * self.nskipped / self.nreceived)
        else:
            skipped_perc = 0

        def fps(x):
            return '%.1f fps' % (x / delta)

        m = ('In the last %.1f s: received %d (%s) processed %d (%s) skipped %d (%s) (%1.f%%)' %
             (delta, self.nreceived, fps(self.nreceived),
              self.nprocessed, fps(self.nprocessed),
              self.nskipped, fps(self.nskipped), skipped_perc))
        return m

if __name__ == "__main__":

    rospy.init_node("pp_lane_controller", anonymous=False)  # adapted to sonjas default file

    pp_lane_controller = pp_lane_controller()
    rospy.spin()
    