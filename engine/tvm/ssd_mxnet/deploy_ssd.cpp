#include <fstream>
#include <vector>
#include <dlpack/dlpack.h>
#include <opencv2/opencv.hpp>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>

using std::vector;

const static float g_mean[3] = {0.485, 0.456, 0.406};
const static float g_std[3] = {0.229, 0.224, 0.225};

int main(void) {

  // Load module
  tvm::runtime::Module mod_dylib =
      tvm::runtime::Module::LoadFromFile(".tvm/deploy_lib.tar.so");

  // Load json graph
  std::ifstream json_in(".tvm/deploy_graph.json", std::ios::in);
  std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
  json_in.close();

  // Load parameters in binary
  std::ifstream params_in(".tvm/deploy_param.params", std::ios::binary);
  std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
  params_in.close();

  TVMByteArray params_arr;
  params_arr.data = params_data.c_str();
  params_arr.size = params_data.length();

  // create runtime
  int device_id = 0;
  int device_type = kDLCPU;
  tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))
        (json_data, mod_dylib, device_type, device_id);

  // get the function from the module(load patameters)
  tvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
  load_params(params_arr);

  // Alloc input tensor
  DLTensor *input;
  int dtype_bits = 32;
  int dtype_lanes = 1;
  int dtype_code = kDLFloat;
  int in_ndim = 4;
  int64_t in_shape[4] = {1, 3, 512, 512};
  TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &input);

  // read image
  cv::Mat image;
  image = cv::imread(".tvm/tmp.jpg");

  // resize image
  cv::Mat resize;
  cv::resize(image, resize, cv::Size(512, 512));

  // conver image to float32
  cv::Mat resize_float;
  resize.convertTo(resize_float, CV_32FC3, 1.0, 0);

  // NHWC -> NCHW
  vector<cv::Mat> split_dst;
  float *input_data = reinterpret_cast<float *>(input->data);
  for (int i = 0; i < 3; ++i) {
    cv::Mat channel(512, 512, CV_32FC1, input_data);
    split_dst.push_back(channel);
    input_data += 512 * 512;
  }
  cv::split(resize_float, split_dst);

  // channel[i] = (channel[i] - mean) / std
  split_dst[0].convertTo(split_dst[0], CV_32FC1, 1.0 / g_std[0], -1.0 * g_mean[0] / g_std[0]);
  split_dst[1].convertTo(split_dst[1], CV_32FC1, 1.0 / g_std[1], -1.0 * g_mean[1] / g_std[1]);
  split_dst[2].convertTo(split_dst[2], CV_32FC1, 1.0 / g_std[2], -1.0 * g_mean[2] / g_std[2]);

  // set input
  tvm::runtime::PackedFunc set_input = mod.GetFunction("set_input");
  set_input("data", input);

  // do inference
  tvm::runtime::PackedFunc run = mod.GetFunction("run");
  run();

  // get output
  tvm::runtime::PackedFunc get_output = mod.GetFunction("get_output");
  tvm::runtime::NDArray res0 = get_output(0);
  tvm::runtime::NDArray res1 = get_output(1);
  tvm::runtime::NDArray res2 = get_output(2);

  return 0;
}
