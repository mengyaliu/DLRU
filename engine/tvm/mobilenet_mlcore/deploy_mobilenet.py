import tvm
import sys
from tvm.contrib import graph_runtime
import numpy as np
from PIL import Image
from utils.timerecorder import TimeRecoder

class Mobilenet():
    def __init__(self, mode, lib_path, graph_path, param_path, class_path):
        # Load compiled lib and params from file
        graph = open(graph_path).read()
        lib = tvm.module.load(lib_path)
        params = bytearray(open(param_path, "rb").read())

        # load classes
        with open(class_path) as f:
            self.classes = eval(f.read())

        # Init target mode
        self.ctx = tvm.cpu(0)
        if mode == 'gpu':
            self.ctx = tvm.gpu(0)

        self.runtime = graph_runtime.create(graph, lib, self.ctx)
        self.runtime.load_params(params)

        # init Timer
        self.timer = TimeRecoder()


    def run(self, im_fname):
        # Load Image
        img = Image.open(im_fname).resize((224, 224))
        img_bgr = np.array(img)[:,:,::-1]
        x = np.transpose(img_bgr, (2, 0, 1))[np.newaxis, :]
        dtype = 'float32'
        tvm_input = tvm.nd.array(x.astype(dtype))
        print(tvm_input.shape)

        # do inference
        self.timer.store_timestamp('inference')
        self.runtime.set_input('image', tvm_input)
        self.runtime.run()
        self.timer.store_timestamp('inference')

        tvm_output = self.runtime.get_output(0)
        top1 = np.argmax(tvm_output.asnumpy()[0])
        results = {'id': str(top1), 'class': self.classes[top1]}
        return results

if __name__ == '__main__':
    mode = sys.argv[1]
    model_dir = sys.argv[2]
    image_path = sys.argv[3]

    lib_path = model_dir + "deploy_lib.tar"
    graph_path = model_dir + "deploy_graph.json"
    param_path = model_dir + "deploy_param.params"
    class_path = model_dir + "classes.txt"

    mobilenet = Mobilenet(mode, lib_path, graph_path, param_path, class_path)
    results = mobilenet.run(image_path)
    print(results)

