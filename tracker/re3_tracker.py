import cv2
import glob
import numpy as np
import os
import tensorflow as tf
import itertools

from skimage.feature import hog
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment

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


class Track:
    def __init__(self, uid, box, image, label, life):
        self.uid = uid
        self.state = [np.zeros((1, LSTM_SIZE)) for _ in range(4)]
        self.box = np.array(box)
        self.image = image
        self.features = None
        self.count = 0
        self.label = label
        self.life = life
        self.age = 0

    @property
    def data(self):
        return self.state, self.box, self.image, self.features, self.count

    def update(self, state, box, image, features, count):
        self.state = state
        self.box = box
        self.image = image
        self.features = features
        self.count = count


class Re3Tracker(object):
    def __init__(self, model_path, iou_threshold, n_init, max_age, gpu_id=GPU_ID):
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

        self.tracks = {}
        self.ids = itertools.count()
        self.iou_threshold = iou_threshold
        self.n_init = n_init
        self.max_age = max_age
        self.total_forward_count = -1


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

    
    def update(self, image, dets, scores, labels):
        """
            :image [np.array[float]] | (h,w,3) | bgr image for a frame
            :dets  [np.array[float]] |  (d,4)  | tlbr detector box per detection
            :scores[np.array[float]] |  (d,)   | detector score per detection
            :labels[list[string]]    |  (d,)   | detector label per detection
        """
        uids = [id_ for id_ in self.tracks]

        if len(uids) == 0:
            for box,label in zip(dets, labels):
                uid = next(self.ids)
                self.tracks[uid] = Track(uid, box, image, label, self.n_init)
            return

        if dets.size == 0:
            rows = []
            columns = []
        else:
            tboxes = [self.tracks[uid].box for uid in uids]
            ious = np.array([[self.iou(tbox, dbox) for dbox in dets] for tbox in tboxes])
            master = ious * (scores / np.mean(scores))
            rows, columns = linear_sum_assignment(-1 * master)

        # check track assignments
        assign = {r: c for r,c in zip(rows, columns)}
        for i, uid in enumerate(uids):
            if i not in assign or master[i, assign[i]] < self.iou_threshold:
                self.tracks[uid].age += 1
                if self.tracks[uid].life < self.n_init or self.tracks[uid].age == self.max_age:
                    del self.tracks[uid]
            else:
                self.tracks[uid].age = 0
                self.tracks[uid].box = dets[assign[i]]
                self.tracks[uid].label = labels[assign[i]]
                self.tracks[uid].life += 1

        # check det assignments
        assign = {c: r for r,c in zip(rows, columns)}
        for i, box in enumerate(dets):
            if i not in assign or uids[assign[i]] not in self.tracks:
                uid = next(self.ids)
                self.tracks[uid] = Track(uid, box, image, labels[i], 1)


    # unique_ids{list{string}}: A list of unique ids for the objects being tracked.
    # image{str or numpy array}: The current image or the path to the current image.
    # starting_boxes{None or dictionary of unique_id to 4x1 numpy array or list}: unique_ids to starting box.
    #    Starting boxes only need to be provided if it is a new track. Bounding boxes in X1, Y1, X2, Y2 format.
    def multi_track(self, image):
        unique_ids = [uid for uid in self.tracks]
        if len(unique_ids) == 0:
            return [], np.zeros((0,0)), []

        image = image.copy()[:,:,::-1]

        # Get inputs for each track.
        images = []
        lstmStates = [[] for _ in range(4)]
        pastBBoxesPadded = []
        for unique_id in unique_ids:
            lstmState, pastBBox, prevImage, originalFeatures, forwardCount = self.tracks[unique_id].data
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
            lstmState, pastBBox, prevImage, originalFeatures, forwardCount = self.tracks[unique_id].data
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
            self.tracks[unique_id].update(lstmState, outputBox, image, originalFeatures, forwardCount)


        mask = [self.tracks[uid].life >= self.n_init for uid in unique_ids]
        uids = [uid for uid, confirmed in zip(unique_ids, mask) if confirmed]
        if len(uids) == 0:
            return [], np.zeros((0,0)), []
        boxes = np.array([box for box, confirmed in zip(outputBoxes, mask) if confirmed])
        return uids, boxes, [self.tracks[uid].label for uid in uids]




