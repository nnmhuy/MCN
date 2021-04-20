import os
import keras
from model.mcn_model import yolo_eval_v2
import numpy as np
from utils.utils import get_one_data_for_testing
from utils.tensorboard_logging import *
import cv2
import keras.backend as K
from matplotlib.pyplot import cm
import spacy
import  progressbar
from google.colab.patches import cv2_imshow
from matplotlib import pyplot as plt

class Evaluate(keras.callbacks.Callback):
    """ Evaluation callback for arbitrary datasets.
    """

    def __init__(
        self,
        image_path,
        query_sentence,
        anchors,
        config,
        tensorboard=None,
        verbose=1
    ):
        """ Evaluate a given dataset using a given model at the end of every epoch during training.

        # Arguments
            generator        : The generator that represents the dataset to evaluate.
            iou_threshold    : The threshold used to consider when a detection is positive or negative.
            score_threshold  : The score confidence threshold to use for detections.
            max_detections   : The maximum number of detections to use per image.
            save_path        : The path to save images with visualized detections to.
            tensorboard      : Instance of keras.callbacks.TensorBoard used to log the mAP value.
            weighted_average : Compute the mAP using the weighted average of precisions among classes.
            verbose          : Set the verbosity level, by default this is set to 1.
        """
        self.image_path = image_path
        self.query_sentence = query_sentence
        self.tensorboard     = tensorboard
        self.verbose         = verbose
        self.batch_size = max(config['batch_size']//2,1)
        self.colors = np.array(cm.hsv(np.linspace(0, 1, 10)).tolist()) * 255
        self.input_shape = (config['input_size'], config['input_size'])  # multiple of 32, hw
        self.config=config
        self.word_embed=spacy.load(config['word_embed'])
        self.word_len = config['word_len']
        self.anchors=anchors
        self.use_nls=config['use_nls']
        # mAP setting
        self.det_acc_thresh = config['det_acc_thresh']
        self.seg_min_overlap=config['segment_thresh']
        if self.tensorboard is not  None:
            self.log_images=config['log_images']
        else:
            self.log_images=0
        self.input_image_shape = K.placeholder(shape=(2,))
        self.sess = K.get_session()
        super(Evaluate, self).__init__()
    def nls(self,pred_seg,pred_box,weight_score=None,lamb_au=-1.,lamb_bu=2,lamb_ad=1.,lamb_bd=0):
        if weight_score is not None:
            #asnls
            mask = np.ones_like(pred_seg, dtype=np.float32)*weight_score*lamb_ad+lamb_bd
            mask[pred_box[1]:pred_box[3] + 1, pred_box[0]:pred_box[2] + 1, ...]=weight_score*lamb_au+lamb_bu
        else:
            #hard-nls
            mask=np.zeros_like(pred_seg,dtype=np.float32)
            mask[pred_box[1]:pred_box[3]+1,pred_box[0]:pred_box[2]+1,...]=1.
        return pred_seg*mask
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        image, seg_image, image_with_seg, result_image = self.evaluate(is_save_images=True)
        logs['image'] = image
        logs['seg_image'] = seg_image
        logs['image_with_seg'] = image_with_seg
        logs['result_image'] = result_image

    def evaluate(self, tag='image', is_save_images=False):
        self.boxes, self.scores, self.eval_inputs = yolo_eval_v2(self.model.output_shape[0],self.anchors, self.input_image_shape,
                                                                               score_threshold=0., iou_threshold=0.)
        # Add the class predict temp dict
        # pred_tmp = []
        seg_prec_all = dict()
        id =0
        seg_iou_all =0.
        detect_prec_all = 0.
        fd_ts_count=0.
        td_fs_count=0.
        fd_fs_count=0.
        # Predict!!!
        images = []
        images_org = []
        files_id = []
        word_vecs = []
        sentences = []
        # gt_segs = []

        image_data, word_vec, image, sentence = get_one_data_for_testing(self.image_path, self.query_sentence, self.input_shape,
                                                                                self.word_embed, self.config,
                                                                                train_mode=False)  # box is [1,5]
        sentences.extend(sentence)
        word_vecs.extend(word_vec)
        # evaluate each sentence corresponding to the same image
        for ___ in range(len(sentence)):
            images.append(image_data)
            images_org.append(image)
            files_id.append(id)
            # gt_segs.append(seg_map)
            id += 1

        images = np.array(images)
        word_vecs = np.array(word_vecs)
        out_bboxes_1, pred_segs,_ = self.model.predict_on_batch([images, word_vecs])
        pred_segs = self.sigmoid_(pred_segs)  # logit to sigmoid
        for i, out in enumerate(out_bboxes_1):
            # Predict
            out_boxes, out_scores = self.sess.run(  # out_boxes is [1,4]  out_scores is [1,1]
                [self.boxes, self.scores],
                feed_dict={
                    # self.eval_inputs: out
                    self.eval_inputs[0]: np.expand_dims(out, 0),
                    self.input_image_shape: np.array(self.input_shape),
                    K.learning_phase(): 0
                })

            pred_box = self.box_value_fix(out_boxes[0], self.input_shape)
            score = out_scores[0]

            # ih = gt_segs[i].shape[0]
            # iw = gt_segs[i].shape[1]
            # w, h = self.input_shape
            # scale = min(w / iw, h / ih)
            # nw = int(iw * scale)
            # nh = int(ih * scale)
            # dx = (w - nw) // 2
            # dy = (h - nh) // 2

            # up sample
            pred_seg = cv2.resize(pred_segs[i], self.input_shape)
            #nls
            if self.use_nls:
                pred_seg = self.nls(pred_seg, self.box_value_fix(out_boxes[0],self.input_shape), out_scores[0])
            #scale to the size of ground-truth
            # pred_seg = pred_seg[dy:nh + dy, dx:nw + dx, ...]
            # pred_seg = cv2.resize(pred_seg, (gt_segs[i].shape[1], gt_segs[i].shape[0]))
            pred_seg = np.reshape(pred_seg, [pred_seg.shape[0], pred_seg.shape[1], 1])

            #visualization
            if is_save_images:
                left, top, right, bottom = pred_box
                # Draw image
                image = np.array(images[i] * 255.).astype(np.uint8)
                # segement image for saving
                seg_image = np.array(
                    cv2.resize(np.array(pred_segs[i] <= self.seg_min_overlap).astype(np.float32),
                                self.input_shape)).astype(
                    np.uint8) * 255
                label = '{:%.2f}' % score
                color = self.colors[0]
                cv2.rectangle(image, (left, top), (right, bottom), color, 2)

                font_size = 0.8

                cv2.putText(image,
                            label,
                            (left, max(top - 3, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_size, color, 2)
                # cv2.putText(image,
                #             str(sentences[i]),
                #             (20, 20),
                #             cv2.FONT_HERSHEY_SIMPLEX,
                #             .9, self.colors[2], 2)
                referring_exp = str(sentences[i])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                seg_image = cv2.cvtColor(seg_image, cv2.COLOR_GRAY2RGB)
                # print(seg_image.shape)
                H, W, C = seg_image.shape
                for i in range(H):
                    for j in range(W):
                        if (np.sum(seg_image[i, j,:]) == 0):
                            seg_image[i,j,1] = 255
                image_with_seg = cv2.addWeighted(image, 0.7, seg_image, 0.3, 0)

                result_image = np.concatenate((image, seg_image, image_with_seg), axis=1)
                result_image = np.concatenate((result_image, np.zeros((100, W * 3, C), dtype=np.uint8)), axis=0)
                cv2.putText(result_image,
                    referring_exp,
                    (20, H + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    .9, self.colors[2], 2)

                return image, seg_image, image_with_seg, result_image

    def sigmoid_(self,x):
        return 1. / (1. + np.exp(-x))
    def box_value_fix(self,box,shape):
        '''
        fix box to avoid numeric overflow
        :param box:
        :param shape:
        :return:
        '''
        top, left, bottom, right = box
        new_w, new_h = shape
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(new_h, np.floor(bottom + 0.5).astype('int32'))
        right = min(new_w, np.floor(right + 0.5).astype('int32'))
        box=np.array([left, top, right, bottom]).astype('int32')
        return box
