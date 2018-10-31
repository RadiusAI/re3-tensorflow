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
    def __init__(self, box, label, life, score):
        self.box = np.array(box)
        self.label = label
        self.life = life
        self.score = score

        self.features = None
        self.state = [np.zeros((1, LSTM_SIZE)) for _ in range(4)]
        self.count = 0
        self.age = 0


class Re3Tracker(object):
    def __init__(self, model_path, iou_threshold, n_init, max_age, gpu_id=GPU_ID):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        tf.Graph().as_default()
        self.image_holder = tf.placeholder(tf.uint8, shape=(None, CROP_SIZE, CROP_SIZE, 3))
        self.lstms_holder = tuple([tf.placeholder(tf.float32, shape=(None, LSTM_SIZE)) for _ in range(4)])
        self.batch_size = tf.placeholder(tf.int32, shape=())
        self.outputs, self.state1, self.state2 = network.inference(
            self.image_holder, num_unrolls=1, batch_size=self.batch_size, train=False, prevLstmState=self.lstms_holder)
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
        self.prev_image = None

    @staticmethod
    def iou(a, b):
        a = a[:, :, np.newaxis]
        b = b.T[np.newaxis]

        lt = np.maximum(a[:, :2], b[:, :2])
        rb = np.minimum(a[:, 2:], b[:, 2:])
        intersect = np.prod(np.maximum(rb - lt, 0), axis=1)

        a_area = np.prod(a[:, 2:] - a[:, :2], axis=1)
        b_area = np.prod(b[:, 2:] - b[:, :2], axis=1)
        union = a_area + b_area - intersect

        return intersect / union

    def update(self, image, dets, scores, labels):
        """
            :image [np.array[float]] | (h,w,3) | bgr image for a frame
            :dets  [np.array[float]] |  (d,4)  | tlbr detector box per detection
            :scores[np.array[float]] |  (d,)   | detector score per detection
            :labels[list[string]]    |  (d,)   | detector label per detection
        """
        tracks = list(self.tracks.items())

        rows, columns = [], []
        if dets.size > 0 and len(tracks) > 0:
            tboxes = np.array([track.box for _, track in tracks])
            master = self.iou(tboxes, dets)
            rows, columns = linear_sum_assignment(-master)

        # check track assignments
        assigns = dict(zip(rows, columns))
        for i, (uid, track) in enumerate(tracks):
            track.age += 1
            j = assigns.get(i, -1)
            if j != -1 and master[i, j] > self.iou_threshold:
                self.tracks[uid] = Track(dets[j], labels[j], track.life + 1, scores[j])
            elif track.life < self.n_init or track.age >= self.max_age:
                del self.tracks[uid]

        # check det assignments
        assigns = dict(zip(columns, rows))
        for i, box in enumerate(dets):
            if i not in assigns or master[assigns[i], i] <= self.iou_threshold:
                uid = next(self.ids)
                self.tracks[uid] = Track(box, labels[i], 0, scores[i])

    # unique_ids{list{string}}: A list of unique ids for the objects being tracked.
    # image{str or numpy array}: The current image or the path to the current image.
    # starting_boxes{None or dictionary of unique_id to 4x1 numpy array or list}: unique_ids to starting box.
    #    Starting boxes only need to be provided if it is a new track. Bounding boxes in X1, Y1, X2, Y2 format.
    def multi_track(self, image):
        image = image.copy()[:, :, ::-1]
        tracks = list(self.tracks.items())
        if len(tracks) == 0:
            self.prev_image = image
            return {}

        # Get inputs for each track.
        crops = [None] * (2 * len(tracks))
        past_boxes = [None] * len(tracks)
        lstm_states = [[None] * len(tracks) for _ in range(4)]
        for i, (uid, track) in enumerate(tracks):
            lstm_state, box = track.state, track.box
            past_crop, past_box = im_util.get_cropped_input(self.prev_image, box, CROP_PAD, CROP_SIZE)
            curr_crop, _ = im_util.get_cropped_input(image, box, CROP_PAD, CROP_SIZE)

            past_boxes[i] = past_box
            crops[2*i], crops[2*i+1] = past_crop, curr_crop
            for ss, state in enumerate(lstm_state):
                lstm_states[ss][i] = state.squeeze()
        self.prev_image = image
        lstm_states = [np.array(states) for states in lstm_states]

        feed_dict = {
            self.image_holder: crops,
            self.lstms_holder: lstm_states,
            self.batch_size: len(tracks)
        }
        raw_output, s1, s2 = self.sess.run([self.outputs, self.state1, self.state2], feed_dict=feed_dict)
        for i, (uid, track) in enumerate(tracks):
            lstm_state = [s1[0][[i], :], s1[1][[i], :], s2[0][[i], :], s2[1][[i], :]]
            features = lstm_state if track.count else track.features

            output_box = bb_util.from_crop_coordinate_system(raw_output[i, :].squeeze() / 10.0, past_boxes[i], 1, 1)
            if track.count > 0 and track.count % MAX_TRACK_LENGTH == 0:
                crop, _ = im_util.get_cropped_input(image, output_box, CROP_PAD, CROP_SIZE)
                input_ = np.tile(crop[np.newaxis, ...], (2, 1, 1, 1))
                feed_dict = {
                    self.image_holder: crop,
                    self.lstms_holder: features,
                    self.batch_size: 1
                }
                _, s1_new, s2_new = self.sess.run([self.outputs, self.state1, self.state2], feed_dict=feed_dict)
                lstm_state = [s1_new[0], s1_new[1], s2_new[0], s2_new[1]]

            track.state = lstm_state
            track.box = output_box
            track.features = features
            track.count += 1

        return {uid: [track.box, track.label, track.score] for uid, track in tracks}
