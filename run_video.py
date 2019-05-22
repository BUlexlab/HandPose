from utils import detector_utils as detector_utils
from utils import pose_classification_utils as classifier
import cv2
import tensorflow as tf
import multiprocessing
from multiprocessing import Queue, Pool
import time
from utils.detector_utils import WebcamVideoStream
import datetime
# import argparse
import os; 
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
import numpy as np

def load_model():
    try:
        model, classification_graph, session = classifier.load_KerasGraph("cnn/models/hand_poses_wGarbage_100.h6")
    except Exception as e:
        print(e)
    detection_graph, sess = detector_utils.load_inference_graph()
    sess = tf.Session(graph=detection_graph)
    return model, classification_graph, detection_graph, sess, session
# if __name__ == '__main__':
def get_hand_info(video, model, classification_graph, detection_graph, sess, session):
    frame_processed = 0
    score_thresh = 0.27
    num_hands = 2
    fps = 1
    width = 300
    height = 200

    # video_capture = WebcamVideoStream(
    #     src=args.video_source, width=args.width, height=args.height).start()

    video_capture = cv2.VideoCapture(video)
    width = 0
    height = 0
    if video_capture.isOpened(): 
        width = int(video_capture.get(3))  # float
        height = int(video_capture.get(4)) # float
    cap_params = {}
    frame_processed = 0
    
    cap_params['im_width'], cap_params['im_height'] = width, height


    print(cap_params['im_width'], cap_params['im_height'])
    cap_params['score_thresh'] = score_thresh

    # max number of hands we want to detect/track
    cap_params['num_hands_detect'] = num_hands

    # print(cap_params, args)
 

    index = 0

    handedness = []
    rhandpoints = []
    lhandpoints = []
    rhand_pred = []
    lhand_pred = []
    ret, frame = video_capture.read()
    while ret:
        ret, frame = video_capture.read()
        frame = cv2.flip(frame, 1)
        index += 1
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except:
            pass
        if (frame is not None):
            # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
            # while scores contains the confidence for each of these boxes.
            # Hint: If len(boxes) > 1 , you may assume you have found atleast one hand (within your score threshold)
            
            boxes, scores = detector_utils.detect_objects(
                frame, detection_graph, sess)
            # print(boxes)
            # print("those were boxes")
            # print(scores)
            # print("those were scores")
            # get region of interest
            boxed, points = detector_utils.get_box_image(cap_params['num_hands_detect'], cap_params["score_thresh"],
                scores, boxes, cap_params['im_width'], cap_params['im_height'], frame)
            print(len(boxed))
            handedness.append(len(boxed))
            if len(boxed) == 1:
                lhandpoints.append(None)
                lhand_pred.append(None)
            if len(boxed) == 0:
                lhandpoints.append(None)
                lhand_pred.append(None)
                rhandpoints.append(None)
                rhand_pred.append(None)
            for c,res in enumerate(boxed):
                # print(points[c])
                # inferences = None
                # classify hand pose
                if res is not None:
                    class_res = classifier.classify(model, classification_graph, session, res)
                    # print(class_res)
                    # inferences_q.put(class_res)
                    if c == 0:
                        rhandpoints.append(np.array(points[c]))
                    if c == 1:
                        lhandpoints.append(np.array(points[c]))
                    try:
                        inferences = list(class_res)
                        # print(inferences)
                        # print(c)
                        if c == 0:
                            rhand_pred.append(inferences)
                        if c == 1:
                            lhand_pred.append(inferences)
                    except Exception as e:
                        # print("ECTEPTIONS")
                        if c == 0:
                            rhand_pred.append(None)
                        if c == 1:
                            lhand_pred.append(None)
                        pass

    return handedness, rhandpoints, lhandpoints, rhand_pred, lhand_pred

# model, classification_graph, detection_graph, sess, session = load_model()
# h, rh, lh, rp, lp = get_hand_info("../highqualityvideos/AGGRESSIVE.mp4", model, classification_graph, detection_graph, sess, session)

# print(h, rh, lh, rp, lp)
# print(rp)
