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
        self.crop = im_util.get_cropped_input(image, box, CROP_PAD, CROP_SIZE)
        self.features = None
        self.count = 0
        self.label = label
        self.life = life
        self.age = 0

    @property
    def data(self):
        return self.state, self.box, self.crop, self.features, self.count

    def update(self, state, box, image, features, count):
        self.state = state
        self.box = box
        self.crop = im_util.get_cropped_input(image, box, CROP_PAD, CROP_SIZE)
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
        a = a[:,:,np.newaxis]
        b = b.T[np.newaxis]

        lt = np.maximum(a[:,:2], b[:,:2])
        rb = np.minimum(a[:,2:], b[:,2:])
        intersect = np.prod(np.maximum(rb - lt, 0), axis=1)

        a_area = np.prod(a[:,2:] - a[:,:2], axis=1)
        b_area = np.prod(b[:,2:] - b[:,:2], axis=1)
        union = a_area + b_area - intersect

        return intersect / union

    
    def update(self, image, dets, scores, labels):
        """
            :image [np.array[float]] | (h,w,3) | bgr image for a frame
            :dets  [np.array[float]] |  (d,4)  | tlbr detector box per detection
            :scores[np.array[float]] |  (d,)   | detector score per detection
            :labels[list[string]]    |  (d,)   | detector label per detection
        """
        image = image[:, :, ::-1]
        tracks = list(self.tracks.items())

        if len(tracks) == 0:
            for box,label in zip(dets, labels):
                uid = next(self.ids)
                self.tracks[uid] = Track(uid, box, image, label, self.n_init)
            return

        if dets.size == 0:
            rows = []
            columns = []
        else:
            tboxes = np.array([track.box for _, track in tracks])
            ious = self.iou(tboxes, dets)
            master = ious * (scores / np.mean(scores))
            rows, columns = linear_sum_assignment(-1 * master)

        # check track assignments
        assign = dict(zip(rows, columns))
        for i, (uid, track) in enumerate(tracks):
            if i not in assign or master[i, assign[i]] < self.iou_threshold:
                track.age += 1
                if track.life < self.n_init or track.age >= self.max_age:
                    del self.tracks[uid]
            else:
                track.age = 0
                track.box = dets[assign[i]]
                track.state = [np.zeros((1, LSTM_SIZE)) for _ in range(4)]
                track.label = labels[assign[i]]
                track.life += 1

        # check det assignments
        assign = dict(zip(columns, rows))
        for i, box in enumerate(dets):
            if i not in assign or master[assign[i], i] < self.iou_threshold:
                uid = next(self.ids)
                self.tracks[uid] = Track(uid, box, image, labels[i], 1)


    # unique_ids{list{string}}: A list of unique ids for the objects being tracked.
    # image{str or numpy array}: The current image or the path to the current image.
    # starting_boxes{None or dictionary of unique_id to 4x1 numpy array or list}: unique_ids to starting box.
    #    Starting boxes only need to be provided if it is a new track. Bounding boxes in X1, Y1, X2, Y2 format.
    def multi_track(self, image):
        tracks = list(self.tracks.items())
        if len(tracks) == 0:
            return {}

        image = image.copy()[:,:,::-1]

        # Get inputs for each track.
        images = [None] * (2 * len(tracks))
        lstmStates = [[None] * len(tracks) for _ in range(4)]
        pastBBoxesPadded = [None] * len(tracks)
        for i, (uid, track) in enumerate(tracks):
            lstmState, pastBBox, prevCrop, originalFeatures, forwardCount = track.data
            croppedInput0, pastBBoxPadded = prevCrop
            croppedInput1, _ = im_util.get_cropped_input(image, pastBBox, CROP_PAD, CROP_SIZE)
            pastBBoxesPadded[i] = pastBBoxPadded
            images[2*i] = croppedInput0
            images[2*i+1] = croppedInput1
            for ss,state in enumerate(lstmState):
                lstmStates[ss][i] = state.squeeze()

        lstmStateArrays = [np.array(state) for state in lstmStates]

        feed_dict = {
            self.imagePlaceholder : images,
            self.prevLstmState : lstmStateArrays,
            self.batch_size : len(images) / 2
        }

        rawOutput, s1, s2 = self.sess.run([self.outputs, self.state1, self.state2], feed_dict=feed_dict)
        for uu, (uid, track) in enumerate(tracks):
            lstmState, _, _, originalFeatures, forwardCount = track.data
            lstmState = [s1[0][[uu],:], s1[1][[uu],:], s2[0][[uu],:], s2[1][[uu],:]]
            if forwardCount == 0:
                originalFeatures = [s1[0][[uu],:], s1[1][[uu],:], s2[0][[uu],:], s2[1][[uu],:]]

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

            track.update(lstmState, outputBox, image, originalFeatures, forwardCount+1)

        return {uid: [track.box, track.label, track.age] for uid, track in tracks}
