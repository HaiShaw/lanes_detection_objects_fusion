#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# @Author  : HaiShaw
# @Site    : https://github.com/HaiShaw
# @File    : lanefusion_node.py


import time
import math
import tensorflow as tf
import numpy as np
import cv2

import copy
from collections import deque as dq

from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess
from config import global_config

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
from lane_fusion.msg import LaneObstacle
from autoware_msgs.msg import DetectedObject
from autoware_msgs.msg import DetectedObjectArray
import message_filters

CFG = global_config.cfg


#####################################################################
# class to keep track the characteristics of each Lane line detection
# N.B.
#    First Implementation, use this class just as global data struct
#    Future TODO: refactor code towards OOP, add methods and classes
#
#    This class is vital to reject bad sample points and fitted line
#
#    Initial Version: have ZERO class parameters!
#    Set it with short memory of 10 last entries.
#
#####################################################################

class Laneline():
    def __init__(self):
        # cache size, tunable:
        self.n = 8
        
        # xbase shift limit (unit: pixel) for rejection
        # self.xbase_offlimit = 100 # 300

        # quadratic coefficient c2 (x-axis intercept) offlimit to reject!
        # c2 is x-axis intercept of fitted curve line, set as ~5% of Xmax
        self.c2fit_offlimit = 600  # in pix
        
        # x base values of the last n fits of the line
        # append use: list_name.append(value)
        # to pop use: list_name.pop(0)
        # self.all_xbases = []
        
        #average x base of the fitted line over the last n iterations
        # self.mean_xbase = int(np.mean(self.all_xbases))
        # self.mean_xbase = None

        # quadratic polynomial coefficients of the last n lines fits
        # e.g. [ array([ 0.00275482,  0.03030303,  0.33333333]),
        #        array([ 0.00275532,  0.03030633,  0.33333465]) ]
        self.all_fits = []
        
        # polynomial coefficient averaged over the last n iterations
        # self.mean_fit = np.mean(self.all_fits, axis=0)
        # e.g. array([ 0.00275482,  0.03030303,  0.33333333])
        self.mean_fit = None

        # X/Y coordinates of last found points from sliding windows
        # self.lastx = None
        # self.lasty = None

        
        ###################################
        #     Undefined -or- obsoleted    #
        #   Some of these can be used to  #
        #  enhance tracking in the future #
        ###################################
        
        # was the line detected in the last iteration?
        # If not, then rejected it and use prior cache
        # N.B. before n good ones cached will not tell
        # self.detected = False

        # x top (where y=0) values of last n fit line
        # x top is calculated post polynomial fit y=0
        # it is good scalar to tell if fit an outlier
        # Initially not used, add support when needed
        # self.all_xtops = []
        
        #average x top of the fitted line over the last n iterations
        # self.mean_xtop = int(np.mean(self.mean_xtop))
        # Initially not used, add support when needed
        # self.mean_xtop = None
        
        # This is obsoleted, as it is the last one in list: all_fits
        # polynomial coefficients for the most recent fit
        # self.current_fit = [np.array([False])]

        # obsolete: last_xbase = all_xbases[-1]
        # self.last_xbase = None
        
        # radius of curvature of the fitted line in some units
        # self.radius_of_curvature = None
        
        # distance in centimeters from vehicle center to lane center
        # self.line_base_pos = None
        
        # difference in fit coefficients between last and new fits
        # self.diffs = np.array([0,0,0], dtype='float')
        
        # x values for detected line pixels
        # self.allx = None
        
        # y values for detected line pixels
        # self.ally = None
        
    """
    # class method to tell if new_xbase is valid one or not
    # to reject noise, it also updates cache if appropriate
    def xbase_valid(self, new_xbase):
        if len(self.all_xbases) < self.n:
            # just add, if not enough history
            self.all_xbases.append(new_xbase)
            self.mean_xbase = int(np.mean(self.all_xbases))
            return True
        else:
            # when this class instance has cached enough hist. records
            # if abs(new_xbase - self.mean_xbase) > self.xbase_offlimit:
            if abs(new_xbase - self.all_xbases[-1]) > self.xbase_offlimit:
                # when new_xbase is offlimit, user should call:
                # xbase_get() to get last good xbase from cache
                return False
            else:
                # update cache and mean
                self.all_xbases.pop(0)
                self.all_xbases.append(new_xbase)
                self.mean_xbase = int(np.mean(self.all_xbases))
                return True

    # class method for user to get last good xbase in cache
    def xbase_get(self):
        if len(self.all_xbases):
            return self.all_xbases[-1]
    """

    # class method to tell if a new fit is valid one or not
    # to reject noise, it also updates cache if appropriate
    # N.B. Input: new_fit is np.array return of np.polyfit()
    def fit_valid(self, new_fit):
        if len(self.all_fits) < self.n:
            # just add, if not enough history
            self.all_fits.append(new_fit)
            self.mean_fit = np.mean(self.all_fits, axis=0)
            return True
        else:
            # when this class instance has cached enough fits history
            if ( abs(new_fit[2] - self.mean_fit[2]) > self.c2fit_offlimit ):
                # when new_fit X-intercept is offlimit, user should call:
                # fit_get() to get last good fit (coeeficient) from cache
                return False
            else:
                # update cache and mean
                self.all_fits.pop(0)
                self.all_fits.append(new_fit)
                self.mean_fit = np.mean(self.all_fits, axis=0)
                return True

    # class method for user to get last good fit coefficients
    def fit_get(self):
        if len(self.all_fits):
            ret = copy.deepcopy(self.mean_fit)
            return ret
            #return self.all_fits[-1]



class lanefusion():
    def __init__(self):
        self.image_topic = rospy.get_param('~image_topic')
        self.fusion_topic = "/detection/fusion_tools/objects"   # rospy.get_param('~fusion_topic')
        self.lanefusion_t = "/lane_fusion"
        self.output_image = rospy.get_param('~output_image')
        self.output_lane = rospy.get_param('~output_lane')
        self.weight_path = rospy.get_param('~weight_path')
        self.use_gpu = rospy.get_param('~use_gpu')

        self.init_lanenet()
        self.bridge = CvBridge()

        ## None sync (2 separate msg handlers)
        ## sub_image = rospy.Subscriber(self.image_topic, Image, self.img_callback, queue_size=1)
        ## sub_fused = rospy.Subscriber(self.fusion_topic, DetectedObjectArray, self.objs_callback, queue_size=1)
        image_sub = message_filters.Subscriber(self.image_topic, Image)
        fused_sub = message_filters.Subscriber(self.fusion_topic, DetectedObjectArray)
        tsync_sub = message_filters.ApproximateTimeSynchronizer([image_sub, fused_sub], 32, 0.1, allow_headerless=True)
        tsync_sub.registerCallback(self.sync_cb)

        self.oboi = ['person','bicycle','car','motorbike','bus','truck']  # 'traffic light', 'stop sign' later
        # how to sync up ingest speed with prior image, use simple solution for now:
        # - At entry of img_callback, set lock / flag so subsequent DetectedObjectArray won't enqueue
        # - At exit of img_callback, release the flag
        # - At entry of objs_callback, simply return when sync queue is locked
        # filtered objs
        self.objs = []
        self.sync = False

        self.pub_image = rospy.Publisher(self.output_image, Image, queue_size=4)
        self.pub_lfuse = rospy.Publisher(self.lanefusion_t, LaneObstacle, queue_size=4)
        self.lane_obst = LaneObstacle()
        self.lane_obst.radius = 5000.0     #  (m)
        self.lane_obst.centerdev = 0.0     # (cm)
        self.lane_obst.anglerror = 0.0     #  (o)
        self.lane_obst.dist2obst_e = -1.0  #  (m)
        self.lane_obst.dist2obst_l = -1.0  #  (m)
        self.lane_obst.dist2obst_r = -1.0  #  (m)

        self.h = 600                       #  pix
        self.w = 800                       #  pix

        self.ym_per_pix = 20./600          #  m/pix
        self.xm_per_pix = 3.6/540          #  m/pix
        self.M    = np.array([[-9.75609756e-01, -1.31707317e+00,  7.90243902e+02],
                              [-2.84217094e-16, -2.92682927e+00,  1.17073171e+03],
                              [-5.26327952e-19, -3.29268293e-03,  1.00000000e+00]])
        self.Minv = np.array([[ 3.25000000e-01, -4.50000000e-01,  2.70000000e+02],
                              [ 0.00000000e+00, -3.41666667e-01,  4.00000000e+02],
                              [ 0.00000000e+00, -1.12500000e-03,  1.00000000e+00]])

        self.lanes_image = None
        self.llane_image = None
        self.rlane_image = None

        # L/R lane line instances
        self.lline = Laneline()
        self.rline = Laneline()
        self.q_RDH = dq([], maxlen=10)

        # Set initial ^2 fitting parameters, for ego lane's center
        self.q_fit = dq([], maxlen=5)      # vertical at pixel 400
        self.q_fit.append(np.array([0.0, 0.0, 400.]))

        self.count = 0
    
    def init_lanenet(self):
        '''
        initlize the tensorflow model
        '''

        self.input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
        phase_tensor = tf.constant('test', tf.string)
        net = lanenet.LaneNet(phase=phase_tensor, net_flag='vgg')
        self.binary_seg_ret, self.instance_seg_ret = net.inference(input_tensor=self.input_tensor, name='lanenet_model')

        # self.cluster = lanenet_cluster.LaneNetCluster()
        self.postprocessor = lanenet_postprocess.LaneNetPostProcessor()

        saver = tf.train.Saver()
        # Set sess configuration
        # if self.use_gpu:
        #     sess_config = tf.ConfigProto(device_count={'GPU': 0})
        # else:
        #     sess_config = tf.ConfigProto(device_count={'CPU': 0})
        # sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
        # sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
        # sess_config.gpu_options.allocator_type = 'BFC'

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
        sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
        sess_config.gpu_options.allocator_type = 'BFC'


        self.sess = tf.Session(config=sess_config)
        saver.restore(sess=self.sess, save_path=self.weight_path)

    
    def sync_cb(self, img, da):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(img, "bgr8")
        except CvBridgeError as e:
            print(e)
        # cv2.namedWindow("ss")
        # cv2.imshow("ss", cv_image)
        # cv2.waitKey(0)
        objs = da.objects

        original_img = cv_image.copy()
        resized_image = self.preprocessing(cv_image)
        self.inference_net(resized_image, original_img)

        if self.lanes_image is None:
            return
        ##
        bv3c_l = cv2.warpPerspective(self.llane_image, self.M, (800,600), flags=cv2.INTER_LINEAR)
        ## TODO this - check BLUE similarity vs. SUM only
        ## bv1c_l = np.sum(bv3c_l, axis=-1)
        ## approximate value filter for reasonable bv1c value range, because resize,
        ## perspective view changes all inject noise/interpolate for channel values
        ####### bv1c_lm = bv1c_l > 200                ## masked at sum(r/g/b) > 200
        bv1c_lB = bv3c_l[:,:,0]                       ## B-GR
        bv1c_lm = bv1c_lB > 200
        leftx, lefty = np.nonzero(bv1c_lm)     ## points of ego-L lane line to fit

        bv3c_r = cv2.warpPerspective(self.rlane_image, self.M, (800,600), flags=cv2.INTER_LINEAR)
        ## TODO this - check GREEN similarity vs. SUM only
        ## bv1c_r = np.sum(bv3c_r, axis=-1)
        ## approximate value filter for reasonable bv1c value range, because resize,
        ## perspective view changes all inject noise/interpolate for channel values
        ####### bv1c_rm = bv1c_r > 200                ## masked at sum(r/g/b) > 200
        bv1c_rR = bv3c_r[:,:,2]                       ## BG-R
        bv1c_rm = bv1c_rR > 200
        rightx, righty = np.nonzero(bv1c_rm)   ## points of ego-R lane line to fit

        ## Lot more complex to handle even single side detection, disable for now
        ## if len(lefty) == 0 or len(righty) == 0:
        if len(lefty) == 0 and len(righty) == 0:
            # return
            warp_img = self.lanes_image
        else:
            ## warp back to camera view
            warp_img = self.colorwarp(self.lanes_image, lefty, leftx, righty, rightx, llane=self.lline, rlane=self.rline)

        # An initial upper bound - provided current detection range from range sensors
        dist2obst_ego = 200
        for i in range(len(objs)):
            if objs[i].label in self.oboi:
                label = objs[i].label
                spacex = objs[i].pose.position.x
                spacey = objs[i].pose.position.y
                euclid = np.sqrt(spacex**2 + spacey**2)
                bboxx = objs[i].x
                bboxy = objs[i].y
                bboxw = objs[i].width
                bboxh = objs[i].height

                pos = (spacex, spacey)
                object_lane = abs(self.object2_ego_lane(pos))
                # assign different color to object according to perceived lane localization
                if object_lane >= 2:
                    color = (125, 255, 0)
                elif object_lane == 1:
                    color = (255, 125, 0)
                else:
                    color = (0, 125, 255)
                    # Ego lane: ignore spacey, and simplifed to use line vs. curve model (TODO)
                    if spacex < dist2obst_ego:
                        dist2obst_ego = spacex

                cv2.rectangle(warp_img, (bboxx, bboxy), (bboxx+bboxw, bboxy+bboxh), color, 2)
                cv2.putText(warp_img, label, (bboxx,bboxy-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                if euclid >= 0.1:     # filter out of range vision detections
                    cv2.putText(warp_img, "{0:5.1f}".format(euclid) + "(m)", (bboxx,bboxy+bboxh+12), \
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        if dist2obst_ego < 200:
            self.lane_obst.dist2obst_e = dist2obst_ego    # update only if range detected, otherwise '-1'

        self.pub_lfuse.publish(self.lane_obst)

        out_img_msg = self.bridge.cv2_to_imgmsg(warp_img, "bgr8")
        self.pub_image.publish(out_img_msg)

        debug = False
        if debug:
            self.count += 1
            if self.count % 10 == 0:
                pos = (20, 0)
                print "pos = (20m, 0m)"
                print self.object2_ego_lane(pos)

                pos = (40, 2)
                print "pos = (40m, 2m)"
                print self.object2_ego_lane(pos)

                pos = (40, -2)
                print "pos = (40m, -2m)"
                print self.object2_ego_lane(pos)

                pos = (50, 3)
                print "pos = (50m, 3m)"
                print self.object2_ego_lane(pos)

                pos = (50, -3)
                print "pos = (50m, -3m)"
                print self.object2_ego_lane(pos)

                pos = (60, 4)
                print "pos = (60m, 4m)"
                print self.object2_ego_lane(pos)

                pos = (60, -4)
                print "pos = (60m, -4m)"
                print self.object2_ego_lane(pos)
        return


    def img_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        # cv2.namedWindow("ss")
        # cv2.imshow("ss", cv_image)
        # cv2.waitKey(0)
        self.sync = True

        original_img = cv_image.copy()
        resized_image = self.preprocessing(cv_image)
        self.inference_net(resized_image, original_img)

        if self.lanes_image is None:
            return
        ##
        bv3c_l = cv2.warpPerspective(self.llane_image, self.M, (800,600), flags=cv2.INTER_LINEAR)
        ## TODO this - check BLUE similarity vs. SUM only
        ## bv1c_l = np.sum(bv3c_l, axis=-1)
        ## approximate value filter for reasonable bv1c value range, because resize,
        ## perspective view changes all inject noise/interpolate for channel values
        ####### bv1c_lm = bv1c_l > 200                ## masked at sum(r/g/b) > 200
        bv1c_lB = bv3c_l[:,:,0]                       ## B-GR
        bv1c_lm = bv1c_lB > 200
        leftx, lefty = np.nonzero(bv1c_lm)     ## points of ego-L lane line to fit

        bv3c_r = cv2.warpPerspective(self.rlane_image, self.M, (800,600), flags=cv2.INTER_LINEAR)
        ## TODO this - check GREEN similarity vs. SUM only
        ## bv1c_r = np.sum(bv3c_r, axis=-1)
        ## approximate value filter for reasonable bv1c value range, because resize,
        ## perspective view changes all inject noise/interpolate for channel values
        ####### bv1c_rm = bv1c_r > 200                ## masked at sum(r/g/b) > 200
        bv1c_rR = bv3c_r[:,:,2]                       ## BG-R
        bv1c_rm = bv1c_rR > 200
        rightx, righty = np.nonzero(bv1c_rm)   ## points of ego-R lane line to fit

        ## Lot more complex to handle even single side detection, disable for now
        ## if len(lefty) == 0 or len(righty) == 0:
        if len(lefty) == 0 and len(righty) == 0:
            # return
            warp_img = self.lanes_image
        else:
            ## warp back to camera view
            warp_img = self.colorwarp(self.lanes_image, lefty, leftx, righty, rightx, llane=self.lline, rlane=self.rline)

        ## Post processing: receive bbox, label, lane_id, etc. then render (warp_img).
        for i in range(len(self.objs)):
                label = self.objs[i]['label']
                spacex = self.objs[i]['spacex']
                spacey = self.objs[i]['spacey']
                euclid = np.sqrt(spacex**2 + spacey**2)
                bboxx = self.objs[i]['bboxx']
                bboxy = self.objs[i]['bboxy']
                bboxw = self.objs[i]['bboxw']
                bboxh = self.objs[i]['bboxh']

                pos = (spacex, spacey)
                object_lane = abs(self.object2_ego_lane(pos))
                # assign different color to object according to perceived lane localization
                if object_lane >= 2:
                    color = (125, 255, 0)
                elif object_lane == 1:
                    color = (255, 125, 0)
                else:
                    color = (0, 125, 255)

                cv2.rectangle(warp_img, (bboxx, bboxy), (bboxx+bboxw, bboxy+bboxh), color, 2)
                cv2.putText(warp_img, label, (bboxx,bboxy-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                if euclid >= 0.1:     # filter out of range vision detections
                    cv2.putText(warp_img, "{0:5.1f}".format(euclid) + "(m)", (bboxx,bboxy+bboxh+12), \
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        out_img_msg = self.bridge.cv2_to_imgmsg(warp_img, "bgr8")
        self.pub_image.publish(out_img_msg)
        #self.objs = []
        #self.sync = False

        self.count += 1
        if self.count % 10 == 0:
            pos = (20, 0)
            print "pos = (20m, 0m)"
            print self.object2_ego_lane(pos)

            pos = (40, 2)
            print "pos = (40m, 2m)"
            print self.object2_ego_lane(pos)

            pos = (40, -2)
            print "pos = (40m, -2m)"
            print self.object2_ego_lane(pos)

            pos = (50, 3)
            print "pos = (50m, 3m)"
            print self.object2_ego_lane(pos)

            pos = (50, -3)
            print "pos = (50m, -3m)"
            print self.object2_ego_lane(pos)

            pos = (60, 4)
            print "pos = (60m, 4m)"
            print self.object2_ego_lane(pos)

            pos = (60, -4)
            print "pos = (60m, -4m)"
            print self.object2_ego_lane(pos)
        self.sync = False


    def preprocessing(self, img):
        image = cv2.resize(img, (512, 256), interpolation=cv2.INTER_LINEAR)
        image = image / 127.5 - 1.0
        # cv2.namedWindow("ss")
        # cv2.imshow("ss", image)
        # cv2.waitKey(1)
        return image

    def inference_net(self, img, original_img):
        binary_seg_image, instance_seg_image = self.sess.run([self.binary_seg_ret, self.instance_seg_ret],
                                                        feed_dict={self.input_tensor: [img]})

        postprocess_result = self.postprocessor.postprocess(
            binary_seg_result=binary_seg_image[0],
            instance_seg_result=instance_seg_image[0],
            source_image=original_img
        )
        # mask_image = postprocess_result
        ##mask_image = postprocess_result['mask_image']
        ##mask_image = cv2.resize(mask_image, (original_img.shape[1],original_img.shape[0]),interpolation=cv2.INTER_LINEAR)
        ##mask_image = cv2.addWeighted(original_img, 0.6, mask_image, 5.0, 0)

        self.lanes_image = postprocess_result['source_image']
        self.llane_image = postprocess_result['l_lane_image']
        self.rlane_image = postprocess_result['r_lane_image']


    def minmax_scale(self, input_arr):
        """

        :param input_arr:
        :return:
        """
        min_val = np.min(input_arr)
        max_val = np.max(input_arr)

        output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

        return output_arr


    def objs_callback(self, data):
        if self.sync:
            return
        self.objs = []
        objs = data.objects
        for i in range(len(objs)):
            if objs[i].label in self.oboi:
                obj = {}
                obj['label'] = objs[i].label
                obj['spacex'] = objs[i].pose.position.x
                obj['spacey'] = objs[i].pose.position.y
                obj['bboxx'] = objs[i].x
                obj['bboxy'] = objs[i].y
                obj['bboxw'] = objs[i].width
                obj['bboxh'] = objs[i].height
                self.objs.append(obj)
                #print objs[i].label
                #print objs[i].pose.position
                #print objs[i].x
                #print objs[i].y
                #print objs[i].width
                #print objs[i].height
        return


    ##############################################################
    #
    # Compute detected objects position relative to ego lane, in
    # terms of relative lane id (N.B. only lookahead view as now):
    # Return:
    #     -2: beyond 1 lane to the left
    #     -1: first lane to the left
    #      0: local lane ahead
    #     +1: first lane to the right
    #     +2: beyond 1 lane to the right
    #  Input:
    #    fit: implicitly acquire and use self.q_fit
    #    pos: object position measured via detection in EgoV coord
    #         (pos.x: distance to front, pos.y: distance to left)
    #
    #  Implications: lane width ~= 3.6 (m)
    #  Improvements: TODO ADAS/Vector MAP
    #  Simplification: Ignore EgoV's error in heading angle,
    #                       and center offset, etc. for now.
    #
    ##############################################################
    def object2_ego_lane(self, pos):
        xm_per_pix = self.xm_per_pix
        ym_per_pix = self.ym_per_pix
        # object x, y to ego frame, unit in (m)
        ox, oy = pos
        # Rule: if front obj within 30(m) and 1/3 lane width away, flag it!
        if ox > 0 and ox < 30.0 and abs(oy) < 1.2:
            return 0
        m_fit = sum(self.q_fit)/len(self.q_fit)
        mf_world = copy.deepcopy(m_fit)
        mf_world[0] = m_fit[0]*xm_per_pix/(ym_per_pix**2)
        mf_world[1] = m_fit[1]*xm_per_pix/ym_per_pix
        mf_world[2] = m_fit[2]*xm_per_pix

        # fitting frame <=> EgoV frame
        fx = -1.0*oy
        fy = -1.0*(ox - 20)      # TODO: 20 to hyper tune!

        # projected X at fy using fit
        px = mf_world[0]*fy**2 + mf_world[1]*fy + mf_world[2]

        # est. lane borders at px: assume on ego lane center
        # ignore lane curvature at (fy, px) below, TODO
        lane_lo_l = px - 1.8
        lane_lo_r = px + 1.8
        lane_l1_l = lane_lo_l - 3.6
        lane_r1_r = lane_lo_r + 3.6

        # compare fx vs. lane borders for ret
        if fx <= lane_l1_l:
            ln = -2
        elif fx <= lane_lo_l:
            ln = -1
        elif fx <= lane_lo_r:
            ln = 0
        elif fx <= lane_r1_r:
            ln = +1
        else:
            ln = +2

        return ln


    ##############################################################
    #
    # Curvature Radius and Vehicle Offset wrt. Center of the Lane
    #
    # Input: to be fitted Left and Right Lane line points finding
    #
    # Return: (curvature, centeroffs)
    #         - curvature in meter
    #         - centeroffs in centimeter,
    #           positive: vehicle is right to the lane center
    #           negative: vehicle is left  to the lane center
    # N.B.
    #      return one curvature radius average from L and R lines
    #
    ##############################################################
    def curvature_centeroffs(self, left_fit, leftx, lefty, right_fit, rightx, righty):
        ym_per_pix = self.ym_per_pix
        xm_per_pix = self.xm_per_pix

        if left_fit is not None:
            left_fit_cr = copy.deepcopy(left_fit)
            left_fit_cr[0] = left_fit[0]*xm_per_pix/(ym_per_pix**2)
            left_fit_cr[1] = left_fit[1]*xm_per_pix/ym_per_pix
            left_fit_cr[2] = left_fit[2]*xm_per_pix
        else:
            # utilize linear transform to save a fitting
            left_fit_cr  = np.polyfit( lefty*ym_per_pix, leftx*xm_per_pix, 2 )

        if right_fit is not None:
            right_fit_cr = copy.deepcopy(right_fit)
            right_fit_cr[0] = right_fit[0]*xm_per_pix/(ym_per_pix**2)
            right_fit_cr[1] = right_fit[1]*xm_per_pix/ym_per_pix
            right_fit_cr[2] = right_fit[2]*xm_per_pix
        else:
            # utilize linear transform to save a fitting
            right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

        y_eval = self.h  # 600 - positioned at bottom horizontal line

        left_curverad =  ((1.0 + (2.0*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) \
                               / np.absolute(2.0*left_fit_cr[0])
        left_headingA = np.arctan(2.0*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])/np.pi*180

        right_curverad = ((1.0 + (2.0*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) \
                               / np.absolute(2.0*right_fit_cr[0])
        right_headingA = np.arctan(2.0*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])/np.pi*180

        # Average curature to yield in-parallel Left/Right Lane Line curves :)
        curverad = (left_curverad + right_curverad) / 2.
        headingA = (left_headingA + right_headingA) / 2.

        lane_xleft = left_fit_cr[0]*(y_eval*ym_per_pix)**2 + left_fit_cr[1]*(y_eval*ym_per_pix) + left_fit_cr[2]
        lane_xright = right_fit_cr[0]*(y_eval*ym_per_pix)**2 + right_fit_cr[1]*(y_eval*ym_per_pix) + right_fit_cr[2]

        lane_center = (lane_xleft + lane_xright) / 2.

        w = self.w       # 800
        vehicle_center = (w * xm_per_pix) / 2.

        centeroffs = (vehicle_center - lane_center)*100
        return (curverad, centeroffs, headingA)


    ##############################################################
    #
    # Color warp found Lane lines back to original undistort image
    #
    # Input:
    #
    #   undistort - undistorted image from original camera's space
    #       leftx - X coordinates of to be fitted Left line points
    #       lefty - Y coordinates of to be fitted Left line points
    #      rightx - X coordinates of to be fitted Right line point
    #      righty - Y coordinates of to be fitted Right line point
    #       llane - class Laneline() instance of tracked Left Lane
    #               pass with 'None' or ignore if no line tracking
    #       rlane - class Laneline() instance of tracked RightLane
    #               pass with 'None' or ignore if no line tracking
    #
    # Return:
    #           Color warped image with Lane lines section merged
    #
    ##############################################################
    def colorwarp(self, undistort, leftx, lefty, rightx, righty, llane = None, rlane = None):
        #def colorwarp(self, undistort, leftx, lefty, rightx, righty):
        # Quadratic fit coefficients

        # Left Lane Line
        if len(leftx) == 0:
            if llane is not None:
                l_fit = llane.fit_get()
            if l_fit is None and rlane is not None:
                l_fit = rlane.fit_get()
                if l_fit is not None:
                    l_fit[2] -= 540
        else:
            l_fit = np.polyfit(lefty, leftx, 2)

        # when 'Left' lane line tracked
        if llane is not None and l_fit is not None:
            # GET 'l_fit' from cache instead if it is considered invalid
            if not llane.fit_valid(l_fit):
                l_fit = llane.fit_get()

        if l_fit is None:
            return undistort

        # Right Lane Line
        if len(rightx) == 0:
            if rlane is not None:
                r_fit = rlane.fit_get()
            if r_fit is None and llane is not None:
                r_fit = llane.fit_get()
                if r_fit is not None:
                    r_fit[2] += 540
        else:
            r_fit = np.polyfit(righty, rightx, 2)

        # when 'Right' lane linetracked
        if rlane is not None and r_fit is not None:
            # GET 'r_fit' from cache instead if it is considered invalid
            if not rlane.fit_valid(r_fit):
                r_fit = rlane.fit_get()

        if r_fit is None:
            return undistort

        # Smooth L and R
        smooth_lr = True
        if smooth_lr:
            l_fit_s = llane.fit_get()
            if l_fit_s is not None:
                l_fit = (l_fit + l_fit_s) / 2.0

            r_fit_s = rlane.fit_get()
            if r_fit_s is not None:
                r_fit = (r_fit + r_fit_s) / 2.0

        # N.B. From now on:
        # Fitted Lane lines (L/R) are considered smooth (noise rejected)!
        
        # balance_lr = True
        # if balance_lr:
        #
        #N.B. I found it less appealing to balance L/R curvature in video
        #     due to independent update to Left and Right line (instance)
        #     In other words - async Left and Right line smooth may cause
        #     balancing not effective, SO it is disabled in this function!
        #
        # This average on high order polynomial coefficients is important
        # Implemented to yield in-parallel Left/Right Lane Line curves :)
        m_fit = (l_fit + r_fit)/2.
        self.q_fit.append(m_fit)

        m_fit = sum(self.q_fit)/len(self.q_fit)

        # Generate Lane line points to both Left and Right
        yvals = np.arange(self.h)                                       # shrink
        l_fitx = m_fit[0]*yvals**2 + m_fit[1]*yvals + m_fit[2] - 100    # center
        r_fitx = l_fitx + 200                                           #  band

        # Calculate Lane lines Curvature Radius and Center Departure
        rad, dev, head = self.curvature_centeroffs(l_fit, leftx, lefty, r_fit, rightx, righty)
        self.q_RDH.append(np.array([rad, dev, head]))
        m_RDH = sum(self.q_RDH)/len(self.q_RDH)
        m_rad = m_RDH[0]
        m_dev = m_RDH[1]
        m_head= m_RDH[2]

        # deploy message memebers
        self.lane_obst.radius = m_rad
        self.lane_obst.centerdev = m_dev
        self.lane_obst.anglerror = m_head

        (h, w) = (self.h, self.w)
        warp_zero = np.zeros((h, w), np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the fitted x and y points into usable format for cv2.fillPoly()
        pts_l = np.array([np.transpose(np.vstack([l_fitx, yvals]))])
        pts_r = np.array([np.flipud(np.transpose(np.vstack([r_fitx, yvals])))])
        pts = np.hstack((pts_l, pts_r))

        # Draw the lane onto the warped blank image in GREEN
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the birdeye view color_warp back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.Minv, (warp_zero.shape[1], warp_zero.shape[0])) 

        # Combine the newwarp with the previous undistrorted image
        result = cv2.addWeighted(undistort, 1, newwarp, 0.1, 0)
        # And add text annotations
        cv2.putText(result,"Curvature Radius: " + "{0:8.2f}".format(m_rad) + "  (m)", (200,30), \
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(result,"Center Departure: " + "{0:8.2f}".format(m_dev) + " (cm)", (200,60), \
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(result,"Heading Angle(o): " + "{0:8.2f}".format(m_head) + " (deg)", (200,90), \
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        return result




if __name__ == '__main__':
    # init args
    rospy.init_node('lanefusion_node')
    lanefusion()
    rospy.spin()
