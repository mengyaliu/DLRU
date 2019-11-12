# -*- coding: utf-8 -*-

import tvm
import sys
from tvm.relay.testing.config import ctx_list
from tvm.contrib import graph_runtime
from gluoncv import data
from timerecorder import TimeRecoder

target_list = ctx_list()

# Load Image
im_fname = sys.argv[1]
x, img = data.transforms.presets.ssd.load_test(im_fname, short=512)

# Function of creating TVM runtime and doing inference
def run(graph, lib, params, ctx):
    # TVM runtime
    m = graph_runtime.create(graph, lib, ctx)
    tvm_input = tvm.nd.array(x.asnumpy(), ctx=ctx)
    m.load_params(params)
    # Inference
    m.run(data=tvm_input)
    # get outputs
    class_IDs, scores, bounding_boxs = m.get_output(0), m.get_output(1), m.get_output(2)
    return class_IDs, scores, bounding_boxs

# Load compiled lib and params from file
loaded_json = open(".tvm/deploy_graph.json").read()
loaded_lib = tvm.module.load(".tvm/deploy_lib.tar")
loaded_params = bytearray(open(".tvm/deploy_param.params", "rb").read())

# init Timer
timer = TimeRecoder()

for target, ctx in target_list:
    timer.store_timestamp('inference')
    class_IDs, scores, bounding_boxs = run(loaded_json, loaded_lib, loaded_params, ctx)
    timer.store_timestamp('inference')
    for i in range(0, len(scores.asnumpy().flatten())):
        if scores.asnumpy()[0][i] < 0.9:
            break;
        print("Result", i, ": ",
              class_IDs.asnumpy()[0][i], ";",
              scores.asnumpy()[0][i], ";",
              bounding_boxs.asnumpy()[0][i])

timer.show()
