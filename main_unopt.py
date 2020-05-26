"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser


def ssd_out(frame, result):
    """
    Parse SSD output.
    :param frame: frame from camera/video
    :param result: list contains the data to parse ssd
    :return: person count and frame
    """
    current_count = 0
    for obj in result[0][0]:
        class_id = int(obj[1])
        # Draw bounding box for object when it's probability is more than
        #  the specified threshold
        if class_id == 15 and obj[2] > prob_threshold:
            xmin = int(obj[3] * initial_w)
            ymin = int(obj[4] * initial_h)
            xmax = int(obj[5] * initial_w)
            ymax = int(obj[6] * initial_h)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 55, 255), 1)
            current_count = current_count + 1
    return frame, current_count


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    
    
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    parser.add_argument("-ft", "--frame_threshold", type=float, default=1,
                        help="Frame threshold for detections filtering"
                        "(0.5 by default)")
    return parser



def connect_mqtt():
    # MQTT server environment variables
    HOSTNAME = socket.gethostname()
    IPADDRESS = socket.gethostbyname(HOSTNAME)
    MQTT_HOST = IPADDRESS
    MQTT_PORT = 3001
    MQTT_KEEPALIVE_INTERVAL = 60
    ### TODO: Connect to the MQTT client ###
    # Connect to the MQTT server
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise 
    model_prototxt = os.path.splitext(args.model)[0] + ".prototxt"
    net = cv2.dnn.readNetFromCaffe(model_prototxt, args.model)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV) #INFERENCE_ENGINE)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    ### TODO: Handle the input stream ###
    # Checks for live feed
    if args.input == 'CAM':
        input_stream = 0

    # Checks for input image
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp') :
        single_image_mode = True
        input_stream = args.input

    # Checks for video file
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "Specified input file doesn't exist"

    cap = cv2.VideoCapture(input_stream)

    if input_stream:
        cap.open(args.input)

    if not cap.isOpened():
        log.error("ERROR! Unable to open video source")
    
    
    global initial_w, initial_h, prob_threshold, frame_threshold
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold
    initial_w = cap.get(3)
    initial_h = cap.get(4)
    FRAME_THRES = args.frame_threshold
    
    # Flag for the input image
    single_image_mode = False
    
    # output video for testing
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    people_counter = cv2.VideoWriter( "people_counter.mp4", 0x00000021, fps, (int(initial_w), int(initial_h)), True)
    #cv2.VideoWriter_fourcc(*"AVC1"),
    
    cur_request_id = 0
    last_count = 0
    prev_count, current_count =0,0
    total_count = 0
    start_time = 0
    
    ## assess perf
    det_time= []
    input_capture_time = []
    frame_count =0
    total_start = time.time()
    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        input_capture_start = time.time() 
        flag, frame = cap.read()
        frame_count +=1
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        ### TODO: Pre-process the image as needed ###
        # convert the frame to a blob and pass the blob through the
        # network and obtain the detections
        W = 300
        H = 300
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W,H), 127.5)
        net.setInput(blob)
        input_capture_time.append(time.time()- input_capture_start)
        
        ### TODO: Start asynchronous inference for specified request ###
        inf_start = time.time()
        result = net.forward()
        det_time.append(time.time() - inf_start)
        ### TODO: Extract any desired stats from the results ###
        frame, detected_count = ssd_out(frame, result)
        if detected_count == prev_count:
            count_conf = 0
        else:
            count_conf +=1
                
        if count_conf == FRAME_THRES: 
        #update prev_count and current_count
            prev_count , current_count = current_count, detected_count
            count_conf = 0
            
        ### TODO: Calculate and send relevant information on ###
        ### current_count, total_count and duration to the MQTT server ###
        ### Topic "person": keys of "count" and "total" ###
            # When new person enters the video
        if current_count > last_count:
            start_time = time.time()
            total_count = total_count + current_count - last_count
            client.publish("person", json.dumps({"total": total_count}))
        cv2.putText(frame, str(total_count)+'\t'+ str(current_count),\
                        (15, 15),cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
            
            ### Topic "person/duration": key of "duration" ###
            # Person duration in the video is calculated
        if current_count < last_count:
            duration = int(time.time() - start_time)
                # Publish messages to the MQTT server
            client.publish("person/duration",
                               json.dumps({"duration": duration}))

        client.publish("person", json.dumps({"count": current_count}))
        last_count = current_count
        people_counter.write(frame)
        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)  
        sys.stdout.flush()
        ### TODO: Write an output image if `single_image_mode` ###
        if single_image_mode:
            cv2.imwrite('output_image.jpg', frame)
    print('input capture time: avg', sum(input_capture_time)*1000/len(input_capture_time), 'ms| min ', min(input_capture_time)*1000,'|max ', max(input_capture_time)*1000)
    print('detection time: avg', sum(det_time)*1000/len(det_time), 'ms | min ', min(det_time)*1000,'| max ' , max(det_time)*1000)
    total_time = time.time() - total_start
    with open( 'stats.txt', 'w') as f:
            f.write(str(round(total_time, 1))+'\n')
            f.write(str(frame_count)+'\n')
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()
    

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    
    # Grab command line args
    args = build_argparser().parse_args()
    
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
