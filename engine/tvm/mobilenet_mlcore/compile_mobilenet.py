import tvm
import sys, os
import tvm.relay as relay
from tvm.contrib.download import download_testdata
import coremltools as cm
import numpy as np
from shutil import copyfile

# Load pretrained CoreML model
model_url = 'https://docs-assets.developer.apple.com/coreml/models/MobileNet.mlmodel'
model_file = 'mobilenet.mlmodel'
model_path = download_testdata(model_url, model_file, module='coreml')
mlmodel = cm.models.MLModel(model_path)

# Compile the model on Relay
if sys.argv[1] == 'cpu':
    target = 'llvm'
else:
    target = 'cuda'

shape_dict = {'image': (1, 3, 224, 224)}

mod, params = relay.frontend.from_coreml(mlmodel, shape_dict)

with relay.build_config(opt_level=3):
    graph, lib, params = relay.build(mod,
                                     target,
                                     params=params)

# save compiled models to files
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

# download imagenet name lists
synset_url = ''.join(['https://gist.githubusercontent.com/zhreshold/',
                      '4d0b62f3d01426887599d4f7ede23ee5/raw/',
                      '596b27d23537e5a1b5751d2b0481ef172f58b539/',
                      'imagenet1000_clsid_to_human.txt'])
synset_name = 'imagenet1000_clsid_to_human.txt'
synset_path = download_testdata(synset_url, synset_name, module='data')

copyfile(synset_path, tmp_dir + 'classes.txt')
