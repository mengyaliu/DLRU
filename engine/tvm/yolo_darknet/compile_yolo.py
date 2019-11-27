# -*- coding: utf-8 -*-

import numpy as np
import sys, os
import tvm
from tvm import relay
from tvm.contrib.download import download_testdata
from tvm.relay.testing.darknet import __darknetffi__

# Model name
MODEL_NAME = sys.argv[2]

# Download darknet model files
CFG_NAME = MODEL_NAME + '.cfg'
WEIGHTS_NAME = MODEL_NAME + '.weights'
REPO_URL = 'https://github.com/dmlc/web-data/blob/master/darknet/'
CFG_URL = REPO_URL + 'cfg/' + CFG_NAME + '?raw=true'
WEIGHTS_URL = 'https://pjreddie.com/media/files/' + WEIGHTS_NAME

cfg_path = download_testdata(CFG_URL, CFG_NAME, module="darknet")
weights_path = download_testdata(WEIGHTS_URL, WEIGHTS_NAME, module="darknet")

# Download and Load darknet library
if sys.platform in ['linux', 'linux2']:
    DARKNET_LIB = 'libdarknet2.0.so'
    DARKNET_URL = REPO_URL + 'lib/' + DARKNET_LIB + '?raw=true'
elif sys.platform == 'darwin':
    DARKNET_LIB = 'libdarknet_mac2.0.so'
    DARKNET_URL = REPO_URL + 'lib_osx/' + DARKNET_LIB + '?raw=true'
else:
    err = "Darknet lib is not supported on {} platform".format(sys.platform)
    raise NotImplementedError(err)

lib_path = download_testdata(DARKNET_URL, DARKNET_LIB, module="darknet")

DARKNET_LIB = __darknetffi__.dlopen(lib_path)
net = DARKNET_LIB.load_network(cfg_path.encode('utf-8'), weights_path.encode('utf-8'), 0)

# compile the model
dtype = 'float32'
batch_size = 1
data = np.empty([batch_size, net.c, net.h, net.w], dtype)
shape_dict = {'data': data.shape}
print("Converting darknet to relay functions...")
mod, params = relay.frontend.from_darknet(net, dtype=dtype, shape=data.shape)

if sys.argv[1] == 'cpu':
    target = 'llvm'
    target_host = 'llvm'
else:
    target = 'cuda'
    target_host = 'cuda'

print("Compiling the model...")
with relay.build_config(opt_level=3):
    graph, lib, params = relay.build(mod,
                                     target=target,
                                     target_host=target_host,
                                     params=params)

# save model to files
tmp_dir = sys.argv[3]
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)
path_lib = tmp_dir + "deploy_lib.tar"
path_graph = tmp_dir + "deploy_graph.json"
path_params = tmp_dir + "deploy_param.params"
lib.export_library(path_lib)
with open(path_graph, "w") as fo:
    fo.write(graph)
with open(path_params, "wb") as fo:
    fo.write(relay.save_param_dict(params))

# download and save names
coco_name = 'coco.names'
coco_url = REPO_URL + 'data/' + coco_name + '?raw=true'
coco_path = download_testdata(coco_url, coco_name, module='data')

with open(coco_path) as f:
    content = f.readlines()

names = {'names': [x.strip() for x in content]}
np.save(tmp_dir + 'names.npy', names)

# save classes
last_layer = net.layers[net.n - 1]
classes = {'classes': last_layer.classes}
np.save(tmp_dir + 'classes.npy', classes)
