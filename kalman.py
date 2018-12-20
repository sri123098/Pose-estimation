import argparse
import logging
import time
from math import *

import cv2
import numpy as np

from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

def hsv_histogram_for_window(frame, window):
    # set up the ROI for tracking
    c,r,w,h = window
    roi = frame[r:r+h, c:c+w]
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
    roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
    cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
    return roi_hist
    
def resample(weights):
    n = len(weights)
    indices = []
    C = [0.] + [sum(weights[:i+1]) for i in range(n)]
    u0, j = np.random.random(), 0
    for u in [(u0+i)/n for i in range(n)]:
      while u > C[j]:
          j+=1
      indices.append(j-1)
    return indices

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--zoom', type=float, default=1.0)
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    logger.debug('cam read+')
    cam = cv2.VideoCapture(args.camera)
    ret_val, image = cam.read()
    logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))
    c,r,w,h = -1,-1,-1,-1
    x = 0
    track_window = (0,0,0,0)
    roi_hist = None
    term_crit = None

    while True:
        ret_val, image = cam.read()

        logger.debug('image preprocess+')
        if args.zoom < 1.0:
            canvas = np.zeros_like(image)
            img_scaled = cv2.resize(image, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_LINEAR)
            dx = (canvas.shape[1] - img_scaled.shape[1]) // 2
            dy = (canvas.shape[0] - img_scaled.shape[0]) // 2
            canvas[dy:dy + img_scaled.shape[0], dx:dx + img_scaled.shape[1]] = img_scaled
            image = canvas
        elif args.zoom > 1.0:
            img_scaled = cv2.resize(image, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_LINEAR)
            dx = (img_scaled.shape[1] - image.shape[1]) // 2
            dy = (img_scaled.shape[0] - image.shape[0]) // 2
            image = img_scaled[dy:image.shape[0], dx:image.shape[1]]

        logger.debug('image process+')
        humans = e.inference(image)

        logger.debug('postprocess+')
        h_p = {}
        image, h_p, num_h = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        logger.debug('show+')
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        #img_final = cv2.rectangle(image, (10,10), (50,50), 255, 2)
        image_final = image
        
        #track items
        if((14 in h_p)):
            frame = image
            #get coordinates
            #x = 1
            #len = math.sqrt(math.pow((h_p[2][0]-h_p[3][0]),2) + math.pow((h_p[2][1]-h_p[3][1]),2))
            #print(len)
            #print(h_p[2], h_p[3])
            x = 1
            #image_final = cv2.rectangle(image, (h_p[3][0]-25,h_p[3][1]), (h_p[3][0]+25,h_p[3][1]+50), 255, 2)
            c,r,w,h = h_p[14][0]-10, h_p[14][1]-10, 70, 70
            track_window = (c,r,w,h)
            roi_hist = hsv_histogram_for_window(frame, (c,r,w,h)) # this is provided for y   
            cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
           
            state = np.array([c+w/2,r+h/2,0,0], dtype='float64') # initial position
            kalman = cv2.KalmanFilter(4,2,0)
            kalman.transitionMatrix = np.array([[1., 0., .1, 0.],
                                            [0., 1., 0., .1],
                                            [0., 0., 1., 0.],
                                            [0., 0., 0., 1.]])
            kalman.measurementMatrix = 1. * np.eye(2, 4)
            kalman.processNoiseCov = 1e-5 * np.eye(4, 4)
            kalman.measurementNoiseCov = 1e-3 * np.eye(2, 2)
            kalman.errorCovPost = 1e-1 * np.eye(4, 4)
            kalman.statePost = state

            # tracking
            prediction = kalman.predict()

            # obtain measurement
            measurement = np.dot(kalman.measurementNoiseCov, np.random.randn(2, 1))
            measurement = np.dot(kalman.measurementMatrix, state) + measurement.reshape(-1)
            kalman.correct(measurement)
            process_noise = sqrt(kalman.processNoiseCov[0,0]) * np.random.randn(4, 1)
            test = np.dot(kalman.transitionMatrix, state)
            state = np.dot(kalman.transitionMatrix, state) + process_noise.reshape(-1)
            
            pt = (int(state[0]), int(state[1]))
            break;
                
        
        cv2.imshow('tf-pose-estimation result', image_final)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break
        logger.debug('finished+')

    while True:
        ret_val, image = cam.read()

        logger.debug('image preprocess+')
        if args.zoom < 1.0:
            canvas = np.zeros_like(image)
            img_scaled = cv2.resize(image, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_LINEAR)
            dx = (canvas.shape[1] - img_scaled.shape[1]) // 2
            dy = (canvas.shape[0] - img_scaled.shape[0]) // 2
            canvas[dy:dy + img_scaled.shape[0], dx:dx + img_scaled.shape[1]] = img_scaled
            image = canvas
        elif args.zoom > 1.0:
            img_scaled = cv2.resize(image, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_LINEAR)
            dx = (img_scaled.shape[1] - image.shape[1]) // 2
            dy = (img_scaled.shape[0] - image.shape[0]) // 2
            image = img_scaled[dy:image.shape[0], dx:image.shape[1]]

        logger.debug('image process+')
        humans = e.inference(image)

        logger.debug('postprocess+')
        h_p = {}
        image, h_p, num_h = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        logger.debug('show+')
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        #img_final = cv2.rectangle(image, (10,10), (50,50), 255, 2)
        frame = image
        if(14 not in h_p):
            c,r,w,h = 0,0,0,0
        else:
            c,r,w,h = h_p[14][0]-10, h_p[14][1]-10, 70, 70
            track_window = (c,r,w,h)
        prediction = kalman.predict()
        if c == 0 and r == 0 and w == 0 and h == 0:
            measurement = np.dot(kalman.measurementNoiseCov, np.random.randn(2, 1))
            measurement = np.dot(kalman.measurementMatrix, state) + measurement.reshape(-1)
            pos = (int(prediction[0]), int(prediction[1]))
        else:
            state = np.array([c+w/2,r+h/2,0,0], dtype='float64') # initial position
            measurement = np.dot(kalman.measurementNoiseCov, np.random.randn(2, 1))
            measurement = np.dot(kalman.measurementMatrix, state) + measurement.reshape(-1)
            posterior = kalman.correct(measurement)
            pos = (int(posterior[0]), int(posterior[1]))

        process_noise = sqrt(kalman.processNoiseCov[0,0]) * np.random.randn(4, 1)
        state = np.dot(kalman.transitionMatrix, state) + process_noise.reshape(-1)

        cv2.circle(frame, pos, 5, (0,0,255), -1)
        cv2.rectangle(frame,(int(int(pos[0])-w/2), int(int(pos[1])-h/2)),(int(int(pos[0]+w/2)), int(int(pos[1])+h/2)),(0,255,0),3)        
        
        cv2.imshow('tf-pose-estimation result', frame)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break
        logger.debug('finished+')

    cv2.destroyAllWindows()
