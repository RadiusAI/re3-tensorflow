import cv2
import glob
import numpy as np
import os
import tensorflow as tf
import time
import itertools

from re3 import network

from re3.re3_utils.util import bb_util
from re3.re3_utils.util import im_util
from re3.re3_utils.tensorflow_util import tf_util

# Network Constants
from re3.constants import CROP_SIZE
from re3.constants import CROP_PAD
from re3.constants import LSTM_SIZE
from re3.constants import LOG_DIR
from re3.constants import GPU_ID
from re3.constants import MAX_TRACK_LENGTH

SPEED_OUTPUT = True

class Re3Tracker(object):
    def __init__(self, model_path, gpu_id=GPU_ID, iou_threshold=0.65):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        tf.Graph().as_default()
        self.imagePlaceholder = tf.placeholder(tf.uint8, shape=(None, CROP_SIZE, CROP_SIZE, 3))
        self.prevLstmState = tuple([tf.placeholder(tf.float32, shape=(None, LSTM_SIZE)) for _ in range(4)])
        self.batch_size = tf.placeholder(tf.int32, shape=())
        self.outputs, self.state1, self.state2 = network.inference(
                self.imagePlaceholder, num_unrolls=1, batch_size=self.batch_size, train=False,
                prevLstmState=self.prevLstmState)
        self.sess = tf_util.Session()
        self.sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt is None:
            raise IOError(('Checkpoint model could not be found.'))
        tf_util.restore(self.sess, ckpt.model_checkpoint_path)

        self.tracked_data = {}
        self.track_labels = {}
        self.ids = itertools.count()
        self.iou_threshold = iou_threshold

        self.time = 0
        self.total_forward_count = -1


    # unique_id{str}: A unique id for the object being tracked.
    # image{str or numpy array}: The current image or the path to the current image.
    # starting_box{None or 4x1 numpy array or list}: 4x1 bounding box in X1, Y1, X2, Y2 format.
    def track(self, unique_id, image, starting_box=None):
        start_time = time.time()

        if type(image) == str:
            image = cv2.imread(image)[:,:,::-1]
        else:
            image = image.copy()

        image_read_time = time.time() - start_time

        if starting_box is not None:
            lstmState = [np.zeros((1, LSTM_SIZE)) for _ in range(4)]
            pastBBox = np.array(starting_box) # turns list into numpy array if not and copies for safety.
            prevImage = image
            originalFeatures = None
            forwardCount = 0
        elif unique_id in self.tracked_data:
            lstmState, pastBBox, prevImage, originalFeatures, forwardCount = self.tracked_data[unique_id]
        else:
            raise Exception('Unique_id %s with no initial bounding box' % unique_id)

        croppedInput0, pastBBoxPadded = im_util.get_cropped_input(prevImage, pastBBox, CROP_PAD, CROP_SIZE)
        croppedInput1,_ = im_util.get_cropped_input(image, pastBBox, CROP_PAD, CROP_SIZE)

        feed_dict = {
                self.imagePlaceholder : [croppedInput0, croppedInput1],
                self.prevLstmState : lstmState,
                self.batch_size : 1,
                }
        rawOutput, s1, s2 = self.sess.run([self.outputs, self.state1, self.state2], feed_dict=feed_dict)
        lstmState = [s1[0], s1[1], s2[0], s2[1]]
        if forwardCount == 0:
            originalFeatures = [s1[0], s1[1], s2[0], s2[1]]

        prevImage = image

        # Shift output box to full image coordinate system.
        outputBox = bb_util.from_crop_coordinate_system(rawOutput.squeeze() / 10.0, pastBBoxPadded, 1, 1)

        if forwardCount > 0 and forwardCount % MAX_TRACK_LENGTH == 0:
            croppedInput, _ = im_util.get_cropped_input(image, outputBox, CROP_PAD, CROP_SIZE)
            input = np.tile(croppedInput[np.newaxis,...], (2,1,1,1))
            feed_dict = {
                    self.imagePlaceholder : input,
                    self.prevLstmState : originalFeatures,
                    self.batch_size : 1,
                    }
            rawOutput, s1, s2 = self.sess.run([self.outputs, self.state1, self.state2], feed_dict=feed_dict)
            lstmState = [s1[0], s1[1], s2[0], s2[1]]

        forwardCount += 1
        self.total_forward_count += 1

        if starting_box is not None:
            # Use label if it's given
            outputBox = np.array(starting_box)

        self.tracked_data[unique_id] = (lstmState, outputBox, image, originalFeatures, forwardCount)
        end_time = time.time()
        if self.total_forward_count > 0:
            self.time += (end_time - start_time - image_read_time)
        if SPEED_OUTPUT and self.total_forward_count % 100 == 0:
            print('Current tracking speed:   %.3f FPS' % (1 / (end_time - start_time - image_read_time)))
            print('Current image read speed: %.3f FPS' % (1 / (image_read_time)))
            print('Mean tracking speed:      %.3f FPS\n' % (self.total_forward_count / max(.00001, self.time)))
        return outputBox


    @staticmethod
    def iou(a, b):
        left = max(a[0], b[0])
        right = min(a[2], b[2])
        top = max(a[1], b[1])
        bottom = min(a[3], b[3])
        
        if right < left or bottom < top:
            return 0

        intersect = (right - left) * (bottom - top)
        a_area = (a[3] - a[1]) * (a[2] - a[0])
        b_area = (b[3] - b[1]) * (b[2] - b[0])
        return intersect / float(a_area + b_area - intersect)


    def update(self, image, dets, labels):
        uids = [id_ for id_ in self.tracked_data]
        tboxes = [self.tracked_data[uid][1] for uid in uids]
        ious = np.array([[self.iou(tbox, dbox) for tbox in tboxes] for dbox in dets])

        if ious.size > 0:
            best = np.max(ious, axis=0)
            for i, uid in enumerate(uids):
                if best[i] < self.iou_threshold:
                    del self.tracked_data[uid]
                    del self.track_labels[uid]

        best = np.max(ious, axis=1) if ious.size > 0 else np.zeros((len(dets),))
        for i, box in enumerate(dets):
            if best[i] < self.iou_threshold:
                lstmState = [np.zeros((1, LSTM_SIZE)) for _ in range(4)]
                pastBBox = np.array(box)
                prevImage = image
                originalFeatures = None
                forwardCount = 0
                uid = next(self.ids)
                self.tracked_data[uid] = (lstmState, pastBBox, image, originalFeatures, forwardCount)
                self.track_labels[uid] = labels[i]


    # unique_ids{list{string}}: A list of unique ids for the objects being tracked.
    # image{str or numpy array}: The current image or the path to the current image.
    # starting_boxes{None or dictionary of unique_id to 4x1 numpy array or list}: unique_ids to starting box.
    #    Starting boxes only need to be provided if it is a new track. Bounding boxes in X1, Y1, X2, Y2 format.
    def multi_track(self, image):
        unique_ids = [uid for uid in self.tracked_data]

        if type(image) == str:
            image = cv2.imread(image)[:,:,::-1]
        else:
            image = image.copy()

        # Get inputs for each track.
        images = []
        lstmStates = [[] for _ in range(4)]
        pastBBoxesPadded = []
        for unique_id in unique_ids:
            lstmState, pastBBox, prevImage, originalFeatures, forwardCount = self.tracked_data[unique_id]
            croppedInput0, pastBBoxPadded = im_util.get_cropped_input(prevImage, pastBBox, CROP_PAD, CROP_SIZE)
            croppedInput1,_ = im_util.get_cropped_input(image, pastBBox, CROP_PAD, CROP_SIZE)
            pastBBoxesPadded.append(pastBBoxPadded)
            images.extend([croppedInput0, croppedInput1])
            for ss,state in enumerate(lstmState):
                lstmStates[ss].append(state.squeeze())

        lstmStateArrays = []
        for state in lstmStates:
            lstmStateArrays.append(np.array(state))

        feed_dict = {
                self.imagePlaceholder : images,
                self.prevLstmState : lstmStateArrays,
                self.batch_size : len(images) / 2
                }
        rawOutput, s1, s2 = self.sess.run([self.outputs, self.state1, self.state2], feed_dict=feed_dict)
        outputBoxes = np.zeros((len(unique_ids), 4))
        for uu,unique_id in enumerate(unique_ids):
            lstmState, pastBBox, prevImage, originalFeatures, forwardCount = self.tracked_data[unique_id]
            lstmState = [s1[0][[uu],:], s1[1][[uu],:], s2[0][[uu],:], s2[1][[uu],:]]
            if forwardCount == 0:
                originalFeatures = [s1[0][[uu],:], s1[1][[uu],:], s2[0][[uu],:], s2[1][[uu],:]]

            prevImage = image

            # Shift output box to full image coordinate system.
            pastBBoxPadded = pastBBoxesPadded[uu]
            outputBox = bb_util.from_crop_coordinate_system(rawOutput[uu,:].squeeze() / 10.0, pastBBoxPadded, 1, 1)

            if forwardCount > 0 and forwardCount % MAX_TRACK_LENGTH == 0:
                croppedInput, _ = im_util.get_cropped_input(image, outputBox, CROP_PAD, CROP_SIZE)
                input = np.tile(croppedInput[np.newaxis,...], (2,1,1,1))
                feed_dict = {
                        self.imagePlaceholder : input,
                        self.prevLstmState : originalFeatures,
                        self.batch_size : 1,
                        }
                _, s1_new, s2_new = self.sess.run([self.outputs, self.state1, self.state2], feed_dict=feed_dict)
                lstmState = [s1_new[0], s1_new[1], s2_new[0], s2_new[1]]

            forwardCount += 1
            self.total_forward_count += 1

            outputBoxes[uu,:] = outputBox
            self.tracked_data[unique_id] = (lstmState, outputBox, image, originalFeatures, forwardCount)

        return unique_ids, outputBoxes, [self.track_labels[uid] for uid in unique_ids]




