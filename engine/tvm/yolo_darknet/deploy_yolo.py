# -*- coding: utf-8 -*-

import numpy as np
import sys
import tvm
from tvm.contrib import graph_runtime
import tvm.relay.testing.yolo_detection
import tvm.relay.testing.darknet

class YOLO():
    def __init__(self, mode, type, model_dir):
        self.MODEL_NAME = type
        self.neth = 416
        self.netw = 416
        self.thresh = 0.5
        self.nms_thresh = 0.45
        self.ctx = tvm.cpu(0)
        if mode == 'gpu':
            self.ctx = tvm.gpu(0)

        graph = open(model_dir + "deploy_graph.json").read()
        lib = tvm.module.load(model_dir + "deploy_lib.tar")
        params = bytearray(open(model_dir + "deploy_param.params", "rb").read())
        self.runtime = graph_runtime.create(graph, lib, self.ctx)
        self.runtime.load_params(params)
        self.names = np.load(model_dir + 'names.npy',allow_pickle='TRUE').item()['names']
        self.classes = np.load(model_dir + 'classes.npy',allow_pickle='TRUE').item()['classes']

    def run(self, im_fname):
        data = tvm.relay.testing.darknet.load_image(im_fname, self.netw, self.neth)
        dtype = 'float32'
        self.runtime.set_input('data', tvm.nd.array(data.astype(dtype)))
        self.runtime.run()
        tvm_out = []
        if self.MODEL_NAME == 'yolov2':
            layer_out = {}
            layer_out['type'] = 'Region'
            # Get the region layer attributes (n, out_c, out_h, out_w, classes, coords, background)
            layer_attr = self.runtime.get_output(2).asnumpy()
            layer_out['biases'] = self.runtime.get_output(1).asnumpy()
            out_shape = (layer_attr[0], layer_attr[1]//layer_attr[0],
                         layer_attr[2], layer_attr[3])
            layer_out['output'] = self.runtime.get_output(0).asnumpy().reshape(out_shape)
            layer_out['classes'] = layer_attr[4]
            layer_out['coords'] = layer_attr[5]
            layer_out['background'] = layer_attr[6]
            tvm_out.append(layer_out)

        elif self.MODEL_NAME == 'yolov3':
            for i in range(3):
                layer_out = {}
                layer_out['type'] = 'Yolo'
                # Get the yolo layer attributes (n, out_c, out_h, out_w, classes, total)
                layer_attr = self.runtime.get_output(i*4+3).asnumpy()
                layer_out['biases'] = self.runtime.get_output(i*4+2).asnumpy()
                layer_out['mask'] = self.runtime.get_output(i*4+1).asnumpy()
                out_shape = (layer_attr[0], layer_attr[1]//layer_attr[0],
                             layer_attr[2], layer_attr[3])
                layer_out['output'] = self.runtime.get_output(i*4).asnumpy().reshape(out_shape)
                layer_out['classes'] = layer_attr[4]
                tvm_out.append(layer_out)

        elif self.MODEL_NAME == 'yolov3-tiny':
            for i in range(2):
                layer_out = {}
                layer_out['type'] = 'Yolo'
                # Get the yolo layer attributes (n, out_c, out_h, out_w, classes, total)
                layer_attr = self.runtime.get_output(i*4+3).asnumpy()
                layer_out['biases'] = self.runtime.get_output(i*4+2).asnumpy()
                layer_out['mask'] = self.runtime.get_output(i*4+1).asnumpy()
                out_shape = (layer_attr[0], layer_attr[1]//layer_attr[0],
                             layer_attr[2], layer_attr[3])
                layer_out['output'] = self.runtime.get_output(i*4).asnumpy().reshape(out_shape)
                layer_out['classes'] = layer_attr[4]
                tvm_out.append(layer_out)
                self.thresh = 0.560
        img = tvm.relay.testing.darknet.load_image_color(im_fname)
        _, im_h, im_w = img.shape
        dets = tvm.relay.testing.yolo_detection.fill_network_boxes((self.netw, self.neth), (im_w, im_h), self.thresh, 1, tvm_out)
        tvm.relay.testing.yolo_detection.do_nms_sort(dets, self.classes, self.nms_thresh)
        num = 0
        results = {}
        for det in dets:
            category = -1
            prob = 0.0
            name = ''
            for j in range(self.classes):
                if det['prob'][j] > self.thresh:
                    if category == -1:
                        category = j
                    prob = det['prob'][j]
                    name = self.names[j]
            if category > -1:
                imc, imh, imw = img.shape
                b = det['bbox']
                left = int((b.x-b.w/2.)*imw)
                right = int((b.x+b.w/2.)*imw)
                top = int((b.y-b.h/2.)*imh)
                bot = int((b.y+b.h/2.)*imh)
                result = {"class": name,
                          "score": prob,
                          "bbox": [left, right, top, bot]}
                results[str(num)] = result;
                num = num + 1
        return results

if __name__ == '__main__':
    yolo = YOLO(sys.argv[1], sys.argv[2], sys.argv[3])
    results = yolo.run(sys.argv[4])
    print(results)
