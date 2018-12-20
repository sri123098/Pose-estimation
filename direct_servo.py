import argparse
import logging
import time
from math import *
import serial
import cv2
import numpy as np
from time import sleep

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

Kp = 0.3
Ki = 0.2
Kd = 0.1
#During the execution we changed the values of the Kp, Ki and Kd values and ran for the 
#set of the requested things i.e Kp=0.2 and,3 
#Kp=0.2 and Ki=0.2 
#Optimal values for Kp Ki and Kd ---- 0.6 0.2 and 0.01 


def PID(required_angle, current_angle):
    global integral, error_previous
    error = required_angle - current_angle
    integral = integral + Ki * error
    out = Kp*error + integral + Kd*(error - error_previous)
    error_previous = error
    return out

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
    
    ser1 = serial.Serial('COM5', 9600) # Establish the connection on a specific port
    sleep(7)
    ser1.close()
    ser2 = serial.Serial('COM6', 9600) # Establish the connection on a specific port
    sleep(7)
    ser2.close()
    
    old_angle1,old_angle2,old_angle3,old_angle4 = 0,0,0,0

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
        if((2 in h_p) and (3 in h_p) and (4 in h_p)):
            #get coordinates
            #x = 1
            #len = math.sqrt(math.pow((h_p[2][0]-h_p[3][0]),2) + math.pow((h_p[2][1]-h_p[3][1]),2))
            #print(len)
            #print(h_p[2], h_p[3])
            x = 1
            image_final = cv2.rectangle(image, (h_p[3][0]-25,h_p[3][1]), (h_p[3][0]+25,h_p[3][1]+50), 255, 2)
            c,r,w,h = h_p[3][0]-25, h_p[3][1], 50, 50
            track_window = (h_p[3][0]-25, h_p[3][1], 50, 50)
            roi_hist = hsv_histogram_for_window(image, track_window)
            def particleevaluator(back_proj, particle):
                return back_proj[particle[1],particle[0]]
            
            # hist_bp: obtain using cv2.calcBackProject and the HSV histogram
            # c,r,w,h: obtain using detect_one_face()
            n_particles = 400
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hist_bp = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)	
            init_pos = np.array([c + w/2.0,r + h/2.0], int) # Initial position
            particles = np.ones((n_particles, 2), int) * init_pos # Init particles to init position
            f0 = particleevaluator(hist_bp,init_pos) * np.ones(n_particles) # Evaluate appearance model
            weights = np.ones(n_particles) / n_particles   # weights are uniform (at first)
    
            stepsize = 15
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
        image_final = image
        if((2 in h_p) and (5 in h_p) and (3 in h_p) and (6 in h_p)):
            result1 = atan2(h_p[3][1] - h_p[2][1], h_p[3][0] - h_p[2][0]) - atan2(h_p[5][1] - h_p[2][1], h_p[5][0] - h_p[2][0]);                
            result2 = atan2(h_p[6][1] - h_p[5][1], h_p[6][0] - h_p[5][0]) - atan2(h_p[2][1] - h_p[5][1], h_p[2][0] - h_p[5][0]);                
            
            res = 0
            res2 = 0
            if((int(degrees(result1)) > 0)):
                print(int(degrees(result1)) - 90)
                res = (int(degrees(result1)) - 90)
            if((int(degrees(result1)) < 0)):
                print(180 - (-1)*(int(degrees(result1))) +90)
                res = (180 - (-1)*(int(degrees(result1))) +90)
            print((int(degrees(result2))))
            if((int(degrees(result2)) < 0)):
                print((-1)*(int(degrees(result2))) - 90)
                res2 = ((-1)*(int(degrees(result2))) - 90)
            if((int(degrees(result2)) > 0)):
                print(int(degrees(result2)) - 90)
                res2 = (int(degrees(result2)) - 90)
            
            if(res>0 and res2>0):
                ser1.open()
                err_angle = PID(res, old_angle1)
                ser1.write(bytes([err_angle]))
                old_angle1 = int.from_bytes(ser1.readline(), byteorder='little')
                ser1.write(bytes([err_angle]))
                err_angle = PID(res2, old_angle2)
                ser1.write(bytes([err_angle]))
                old_angle2 = int.from_bytes(ser1.readline(), byteorder='little')
                ser1.close()
       
        if((2 in h_p) and (3 in h_p) and (4 in h_p) and (5 in h_p) and (6 in h_p) and (7 in h_p)):
            result1 = atan2(h_p[2][1] - h_p[3][1], h_p[2][0] - h_p[3][0]) - atan2(h_p[4][1] - h_p[3][1], h_p[4][0] - h_p[3][0]);                
            #result2 = atan2(h_p[6][1] - h_p[5][1], h_p[6][0] - h_p[5][0]) - atan2(h_p[2][1] - h_p[5][1], h_p[2][0] - h_p[5][0]);                
            
            res = 0
            res2 = 0
            print(int(degrees(result1)))
            if((int(degrees(result1)) > 0)):
                print(int(degrees(result1)))
                res = (int(degrees(result1)))
            if((int(degrees(result1)) < 0)):
                print(0)
                res = (0)   
            if(res>0):
                ser2.open()
                ser2.write(bytes([res]))
                ser2.close()
        
        if((2 in h_p) and (3 in h_p) and (4 in h_p) and (5 in h_p) and (6 in h_p) and (7 in h_p)):
            #print("here")
            result1 = atan2(h_p[5][1] - h_p[6][1], h_p[5][0] - h_p[6][0]) - atan2(h_p[7][1] - h_p[6][1], h_p[7][0] - h_p[6][0]);                
            #result2 = atan2(h_p[6][1] - h_p[5][1], h_p[6][0] - h_p[5][0]) - atan2(h_p[2][1] - h_p[5][1], h_p[2][0] - h_p[5][0]);                
            
            res = 0
            res2 = 0
            print(int(degrees(result1)))
            if((int(degrees(result1)) > 0)):
                print(int(degrees(result1)) -180)
                res = (int(degrees(result1)) - 180)
            if((int(degrees(result1)) < 0)):
                print(0)
                res = (0)
            if(res>0 and res2>0):
                ser2.open()
                err_angle = PID(res, old_angle3)
                ser2.write(bytes([err_angle]))
                old_angle3 = int.from_bytes(ser2.readline(), byteorder='little')
                ser2.write(bytes([err_angle]))
                err_angle = PID(res2, old_angle4)
                ser2.write(bytes([err_angle]))
                old_angle4 = int.from_bytes(ser2.readline(), byteorder='little')
                ser2.close()               
        
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break
        logger.debug('finished+')

    cv2.destroyAllWindows()
