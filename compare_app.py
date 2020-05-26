import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

def ssd_out(result, thres):
    """
    Parse SSD output.
    :param frame: frame from camera/video
    :param result: list contains the data to parse ssd
    :return: person count and frame
    """
    current_count = 0
    conf = 0
    for obj in result[0][0]:
        class_id = int(obj[1])
        # Draw bounding box for object when it's probability is more than
        #  the specified threshold
        if class_id == 15 and obj[2]> thres:
            current_count = current_count + 1
            conf = obj[2]
    return current_count, conf

def build_argparser():
    """
    Parse command line arguments. 
    :return: command line arguments 
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required = True, type = str, 
                        help = "Path to trained model" )
    return parser

def infer_OpenCV_unopt(input_stream, model):
    model_prototxt = model + ".prototxt"
    model_caffemodel = model + ".caffemodel"
    
    net = cv2.dnn.readNetFromCaffe(model_prototxt, model_caffemodel)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    ## input_stream
    cap = cv2.VideoCapture(input_stream)
    
    ## video writer
    #people_counter = cv2.VideoWriter("")
    
    ## Performance related variables
    frame_count = 0 # For fps evaluation
    det_time = []
    total_start = time.time()
    min_conf = 1.0
    while cap.isOpened():
        flag, frame = cap.read()
        frame_count +=1
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        w, h = 300, 300 # No get input size in net class of opencv
        # performs all 3 preprocessing steps above + mean and scale adjustment
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (w,h), 127.5)
        net.setInput(blob)
        # Inference step
        inf_start = time.time()
        result = net.forward()
        det_time.append(time.time() - inf_start)
        count,conf = ssd_out(result, 0)
        if count >0 and conf< min_conf : min_conf = conf
        with open('stats_OpenCV_unopt.txt', 'a') as f:
            f.write(str(count)+'\t' + str(conf)+'\n')
    
    total_time = time.time()-total_start
    det_time_avg = sum(det_time)*1000/len(det_time)
    with open('stats_OpenCV_unopt.txt', 'a') as f:
        f.write('*END* \n')
        f.write('Average detection time(ms) \t'+ str(det_time_avg) + '\n')
        f.write('Total_time(s) \t'+ str(round(total_time,1))+ '\n')
        f.write('Frame_count \t'+ str(frame_count)+'\n')
        f.write('Input stream fps \t'+ str(int(cap.get(cv2.CAP_PROP_FPS))))
        f.write('minimum confidence \t'+ str(min_conf))
    cap.release()
    
def infer_IntelIE_opt(input_stream, model):
    
    ## set HW device and model
    net = Network()
    cpu_extension = '/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so'
    
    net.load_model(model+'.xml', 'CPU', cpu_extension)
    n,c, h, w = net.get_input_shape()
    
    ## input_stream
    cap = cv2.VideoCapture(input_stream)
    
    ## Performance related variables
    frame_count = 0 # For fps evaluation
    det_time = []
    total_start = time.time()
    min_conf = 1.0
    # duration related 
    current_count = 0
    last_count = 0
    with open('duration_IntelIE_opt.txt', 'w') as f:
        f.write('Method1 (s) \t Method2 (s) \n')
    
    while cap.isOpened():
        flag, frame = cap.read()
        frame_count +=1
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        
        blob = net.preprocess(frame)
        
        # Inference step
        inf_start = time.time()
        net.execute(0,blob)
        net.wait(0)
        result = net.get_output(0)
        det_time.append(time.time() - inf_start)
        count,conf = ssd_out(result, 0)
        if count >0 and conf< min_conf : min_conf = conf
        with open('stats_IntelIE_opt.txt', 'a') as f:
            f.write(str(count)+ '\t' + str(conf)+'\n')
        # Person duration in the video is calculated
        current_count = count
        if current_count > last_count:
            start_time = time.time()
            start_frame = frame_count
        if current_count< last_count:
            duration1 = round(time.time()-start_time, 1)
            duration2 = (frame_count -start_frame)/int(cap.get(cv2.CAP_PROP_FPS))
            with open('duration_IntelIE_opt.txt', 'a') as f:
                f.write(str(duration1) + '\t'+ str(duration2) + '\n')                                    
        last_count = current_count
    total_time = time.time()-total_start
    det_time_avg = sum(det_time)*1000/len(det_time)
    with open('stats_IntelIE_opt.txt', 'a') as f:
        f.write('*END* \n')
        f.write('Average detection time(ms) \t'+ str(det_time_avg) + '\n')
        f.write('Total_time(s) \t'+ str(round(total_time,1))+ '\n')
        f.write('Frame_count \t'+ str(frame_count)+'\n')
        f.write('Input stream fps \t'+ str(int(cap.get(cv2.CAP_PROP_FPS))))
        f.write('minimum confidence \t'+ str(min_conf))
    cap.release()

def regularize(file, frame_thres):
    f = open(file, 'r')
    frame_count = 0
    last_correct_count = 0
    correct_count = 0
    frame_conf = 0
    for line in f:
        if "END" in line:
            f.close()
            break
        frame_count +=1
        #print('Frame count =', str(frame_count))
        count, conf = line.split()
        detected_count = int(count)
        if  detected_count < last_correct_count:
            # hold on and count
            frame_conf +=1
        else:
            correct_count = detected_count
            # write skeptical frames
            for i in range(frame_conf+1):
                #print(i)
                with open('stats_Ground_Truth.txt','a') as f_reg:
                    f_reg.write(str(correct_count)+ '\t' + str(conf)+'\n')
            
            frame_conf = 0
        if frame_conf == frame_thres:
            correct_count = detected_count
            for i in range(frame_thres):
                #print(i)
                with open('stats_Ground_Truth.txt','a') as f_reg:
                    f_reg.write(str(correct_count)+ '\t' + str(conf)+'\n')
            frame_conf = 0
        last_correct_count = correct_count
    
    f_reg.close()

def calculate_accuracy(file1, file2):
    f1 = open(file1, 'r')
    f2 = open(file2, 'r')
    compared_frame_count = 0
    matched_frame_count = 0
    while (True):
        line1 = f1.readline()
        line2 = f2.readline()
        if "*END*" in line1 or "*END*" in line2:
            break
        compared_frame_count +=1
        count1, conf1 = line1.split()
        count2, conf2 = line2.split()
        if count1 == count2: 
            matched_frame_count +=1
        #else:
            #print ('#Frame {}: {} \t {} \n'.format(compared_frame_count, count1, count2))
    f1.close()
    f2.close()
    print('Matching {} out of {} frames \n'.format(matched_frame_count,compared_frame_count))
    print ('Accuracy = {} %'.format(100* matched_frame_count//compared_frame_count))
    
def main():
    """
    Parse the arguments and infer on different backends

    :return: None
    """
    
    # Grab command line args
    args = build_argparser().parse_args()
    model = os.path.splitext(args.model)[0]
    

    # Perform inference on the input stream
    input_stream = 'resources/Pedestrian_Detect_2_1_1.mp4'
    assert os.path.isfile(input_stream), "Specified input file doesn't exist"
    #infer_OpenCV_unopt(input_stream, model)
    infer_IntelIE_opt(input_stream, model)
    regularize('stats_OpenCV_unopt.txt',10)
    calculate_accuracy('stats_Ground_Truth.txt', 'stats_OpenCV_unopt.txt')
    calculate_accuracy('stats_Ground_Truth.txt', 'stats_IntelIE_opt.txt')
        
    
if __name__ == '__main__':
    main()
