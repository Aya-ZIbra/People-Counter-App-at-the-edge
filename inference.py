#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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
import logging as log
from openvino.inference_engine import IENetwork, IECore
import cv2

class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        pass

    def load_model(self, model, device, cpu_extension = None, num_requests= 1, core = None):
        ### TODO: Load the model ###
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        # Initialize ie
        if not core:
            log.info("Creating Inference Engine")
            self.ie = IECore()
        else:
            self.ie = core
        
        ### TODO: Add any necessary extensions ###
        if cpu_extension and 'CPU' in device:
            self.ie.add_extension(cpu_extension, 'CPU')
            
        
        #creat network
        log.info("Reading IR...")
        self.net = IENetwork(model=model_xml, weights=model_bin)
        
        ### TODO: Check for supported layers ###
        log.info("Checking for support layers...")
        if "CPU" in device:
            supported_layers = self.ie.query_network(self.net, "CPU")
            not_supported_layers = \
            [l for l in self.net.layers.keys() if l not in supported_layers]
            if len(not_supported_layers) != 0:
                log.error("Following layers are not supported by the plugin for specified device {}:\n {}".format(args.device, ', '.join(not_supported_layers)))
                log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
                sys.exit(1)
        
        log.info("Loading model to the plugin")
        self.exec_net = self.ie.load_network(network=self.net, device_name=device)
        #, num_requests = num_requests)
       
        ### TODO: Return the loaded inference plugin ###
        return self.exec_net # may return the plugin later
    
    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        
        self.input_blob = next(iter(self.net.inputs))
        return self.net.inputs[self.input_blob].shape

    def preprocess(self, frame):
        n, c, h, w = self.get_input_shape()
        image = cv2.resize(frame, (w, h))
        # Change data layout from HWC to CHW
        image = image.transpose((2, 0, 1))
        image = image.reshape((n, c, h, w))
        return image
    def execute(self, request_id, frame):
        ### TODO: Start an asynchronous request ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        """
        Starts asynchronous inference for specified request.
        :param request_id: Index of Infer request value. Limited to device capabilities.
        :param frame: Input image
        :return: Instance of Executable Network class
        """
        self.infer_request_handle = self.exec_net.start_async(
            request_id=request_id, inputs={self.input_blob: frame})
        return self.exec_net
        

    def wait(self, request_id):
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        status = self.exec_net.requests[request_id].wait(-1)
        # -1 -> Waits until inference result becomes available (default value) 
        return status

    def get_output(self, request_id):
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        self.output_blob = next(iter(self.net.outputs))
        res = self.exec_net.requests[request_id].outputs[self.output_blob]
        return res
    
    def clean(self):
        """
        Deletes all the instances
        :return: None
        """
        del self.exec_net
        del self.ie
        del self.net