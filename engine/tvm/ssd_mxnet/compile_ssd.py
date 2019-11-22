# -*- coding: utf-8 -*-

import tvm
import sys, os
from tvm.relay.testing.config import ctx_list
from tvm import relay
from gluoncv import model_zoo, data, utils
from tvm.contrib import util

supported_model = [
    'ssd_512_resnet50_v1_voc',
    'ssd_512_resnet50_v1_coco',
    'ssd_512_resnet101_v2_voc',
    'ssd_512_mobilenet1.0_voc',
    'ssd_512_mobilenet1.0_coco',
    'ssd_300_vgg16_atrous_voc'
    'ssd_512_vgg16_atrous_coco',
]

model_name = supported_model[0]
dshape = (1, 3, 512, 512)

if sys.argv[1] == 'cpu':
    target = 'llvm'
    ctx = tvm.cpu(0)
else:
    target = 'cuda'
    ctx = tvm.gpu(0)

# download model
block = model_zoo.get_model(model_name, pretrained=True)

# function of compiling model
def build(target):
    mod, params = relay.frontend.from_mxnet(block, {"data": dshape})
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(mod, target, params=params)
    return graph, lib, params

# compile model and save them to files
graph, lib, params = build(target)
tmp_dir = sys.argv[2]
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
