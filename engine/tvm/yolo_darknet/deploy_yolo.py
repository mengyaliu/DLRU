# -*- coding: utf-8 -*-

import numpy as np
import sys
import tvm
from tvm.contrib import graph_runtime
from ctypes import *
from tvm.contrib.download import download_testdata
from tvm.relay.testing.darknet import __darknetffi__
import tvm.relay.testing.yolo_detection
import tvm.relay.testing.darknet

MODEL_NAME = 'yolov3'

######################################################################
# Load a test image
# -----------------
neth = 416
netw = 416
img_path = sys.argv[3]
data = tvm.relay.testing.darknet.load_image(img_path, netw, neth)

######################################################################
# Execute on TVM Runtime
# ----------------------
# The process is no different from other examples.
mode = sys.argv[1]
target = 'llvm'
ctx = tvm.cpu(0)
if mode == 'gpu':
    target = 'cuda'
    ctx = tvm.gpu(0)

model_dir = sys.argv[2]
graph = open(model_dir + "deploy_graph.json").read()
lib = tvm.module.load(model_dir + "deploy_lib.tar")
params = bytearray(open(model_dir + "deploy_param.params", "rb").read())

m = graph_runtime.create(graph, lib, ctx)
m.load_params(params)

# set inputs
dtype = 'float32'
m.set_input('data', tvm.nd.array(data.astype(dtype)))

# execute
print("Running the test image...")

# detection
# thresholds
thresh = 0.5
nms_thresh = 0.45

m.run()
# get outputs
tvm_out = []
if MODEL_NAME == 'yolov2':
    layer_out = {}
    layer_out['type'] = 'Region'
    # Get the region layer attributes (n, out_c, out_h, out_w, classes, coords, background)
    layer_attr = m.get_output(2).asnumpy()
    layer_out['biases'] = m.get_output(1).asnumpy()
    out_shape = (layer_attr[0], layer_attr[1]//layer_attr[0],
                 layer_attr[2], layer_attr[3])
    layer_out['output'] = m.get_output(0).asnumpy().reshape(out_shape)
    layer_out['classes'] = layer_attr[4]
    layer_out['coords'] = layer_attr[5]
    layer_out['background'] = layer_attr[6]
    tvm_out.append(layer_out)

elif MODEL_NAME == 'yolov3':
    for i in range(3):
        layer_out = {}
        layer_out['type'] = 'Yolo'
        # Get the yolo layer attributes (n, out_c, out_h, out_w, classes, total)
        layer_attr = m.get_output(i*4+3).asnumpy()
        layer_out['biases'] = m.get_output(i*4+2).asnumpy()
        layer_out['mask'] = m.get_output(i*4+1).asnumpy()
        out_shape = (layer_attr[0], layer_attr[1]//layer_attr[0],
                     layer_attr[2], layer_attr[3])
        layer_out['output'] = m.get_output(i*4).asnumpy().reshape(out_shape)
        layer_out['classes'] = layer_attr[4]
        tvm_out.append(layer_out)

elif MODEL_NAME == 'yolov3-tiny':
    for i in range(2):
        layer_out = {}
        layer_out['type'] = 'Yolo'
        # Get the yolo layer attributes (n, out_c, out_h, out_w, classes, total)
        layer_attr = m.get_output(i*4+3).asnumpy()
        layer_out['biases'] = m.get_output(i*4+2).asnumpy()
        layer_out['mask'] = m.get_output(i*4+1).asnumpy()
        out_shape = (layer_attr[0], layer_attr[1]//layer_attr[0],
                     layer_attr[2], layer_attr[3])
        layer_out['output'] = m.get_output(i*4).asnumpy().reshape(out_shape)
        layer_out['classes'] = layer_attr[4]
        tvm_out.append(layer_out)
        thresh = 0.560

# do the detection and bring up the bounding boxes

names = np.load(model_dir + 'names.npy',allow_pickle='TRUE').item()['names']
classes = np.load(model_dir + 'classes.npy',allow_pickle='TRUE').item()['classes']

img = tvm.relay.testing.darknet.load_image_color(img_path)
_, im_h, im_w = img.shape
dets = tvm.relay.testing.yolo_detection.fill_network_boxes((netw, neth), (im_w, im_h), thresh,
                                                      1, tvm_out)
tvm.relay.testing.yolo_detection.do_nms_sort(dets, classes, nms_thresh)

num = 0
results = {}
for det in dets:
    category = -1
    prob = 0.0
    name = ''
    for j in range(classes):
        if det['prob'][j] > thresh:
            if category == -1:
                category = j
            prob = det['prob'][j]
            name = names[j]
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

print(results)
