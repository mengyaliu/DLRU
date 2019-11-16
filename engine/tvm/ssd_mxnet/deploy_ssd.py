# -*- coding: utf-8 -*-

import tvm
import sys
from tvm.relay.testing.config import ctx_list
from tvm.contrib import graph_runtime
from gluoncv import data
from utils.timerecorder import TimeRecoder

class SSD():
    def __init__(self):
        # Load compiled lib and params from file
        loaded_json = open(".tvm/deploy_graph.json").read()
        loaded_lib = tvm.module.load(".tvm/deploy_lib.tar")
        loaded_params = bytearray(open(".tvm/deploy_param.params", "rb").read())

        # Init runtime
        self.target = 'llvm'
        self.ctx = tvm.cpu(0)
        self.runtime = graph_runtime.create(loaded_json, loaded_lib, self.ctx)
        self.runtime.load_params(loaded_params)

        # init Timer
        self.timer = TimeRecoder()

        # init threashold
        self.thresh = 0.9

    def run(self, im_fname):
        # Load Image
        input, img = data.transforms.presets.ssd.load_test(im_fname, short=512)
        tvm_input = tvm.nd.array(input.asnumpy(), ctx=self.ctx)

        # do inference
        self.timer.store_timestamp('inference')
        self.runtime.run(data=tvm_input)
        class_IDs, scores, bounding_boxs = self.runtime.get_output(0), self.runtime.get_output(1), self.runtime.get_output(2)
        self.timer.store_timestamp('inference')

        # get outputs
        results = {}
        for i in range(0, len(scores.asnumpy().flatten())):
            if scores.asnumpy()[0][i] < self.thresh: 
                break;
            result = {"class": class_IDs.asnumpy()[0][i].tolist(),
                      "score": scores.asnumpy()[0][i].tolist(),
                      "bbox": bounding_boxs.asnumpy()[0][i].tolist()}
            results[str(i)] = result;
        self.timer.show()
        return results

if __name__ == '__main__':
    ssd = SSD()
    results = ssd.run(sys.argv[1])
    print(results)
