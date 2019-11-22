#include <sstream>
#include <fstream>

#include "TvmSsdMxnet.hpp"
#include "dlru_utils.hpp"

#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#define THRESH 0.9

using std::endl;
using std::string;
using std::vector;

const static float g_mean[3] = {0.485, 0.456, 0.406};
const static float g_std[3] = {0.229, 0.224, 0.225};

namespace DLRU {

void TvmSsdMxnet::Init(const string &mode,
                       const string &lib,
                       const string &graph,
                       const string &params) {

  // init configurations
  thresh_ = THRESH;

  for(int i = 0; i < 3; i++) {
    mean_.push_back(g_mean[i]);
    std_.push_back(g_std[i]);
  }

  // Load module
  tvm::runtime::Module mod_dylib = tvm::runtime::Module::LoadFromFile(lib);

  // Load json graph
  std::ifstream json_in(graph, std::ios::in);
  string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
  json_in.close();

  // Load parameters in binary
  std::ifstream params_in(params, std::ios::binary);
  string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
  params_in.close();

  TVMByteArray params_arr;
  params_arr.data = params_data.c_str();
  params_arr.size = params_data.length();

  // create runtime
  int device_id = 0;
  int device_type = (!mode.compare(0, 3, "gpu") || !mode.compare(0, 3, "GPU")) ? kDLGPU : kDLCPU;
  LOG(INFO) << "runtime mode: " << mode << ", " << device_type;
  mod_ = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))
                                (json_data, mod_dylib, device_type, device_id);

  // load patameters
  tvm::runtime::PackedFunc load_params = mod_.GetFunction("load_params");
  load_params(params_arr);

  // Alloc input tensor
  int dtype_bits = 32;
  int dtype_lanes = 1;
  int dtype_code = kDLFloat;
  int in_ndim = 4;
  int64_t in_shape[4] = {1, 3, 512, 512};
  TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &input_);

  LOG(INFO) << "DLRU tvm runtime initialized with module file: " << lib;
}

TvmSsdMxnet::TvmSsdMxnet() { }

TvmSsdMxnet::TvmSsdMxnet(const string &mode,
                         const string &lib,
                         const string &graph,
                         const string &params) {
  Init(mode, lib, graph, params);
}

TvmSsdMxnet::~TvmSsdMxnet() {

}

void TvmSsdMxnet::PreProcess(cv::Mat &image) {
  // resize image
  cv::Mat resize;
  cv::resize(image, resize, cv::Size(512, 512));

  // conver image to float32
  cv::Mat resize_float;
  resize.convertTo(resize_float, CV_32FC3, 1.0, 0);

  // NHWC -> NCHW, using input_ tensor
  vector<cv::Mat> split_dst;
  float *input_data = reinterpret_cast<float *>(input_->data);
  for (int i = 0; i < 3; ++i) {
    cv::Mat channel(512, 512, CV_32FC1, input_data);
    split_dst.push_back(channel);
    input_data += 512 * 512;
  }
  cv::split(resize_float, split_dst);

  // channel[i] = (channel[i] - mean) / std
  split_dst[0].convertTo(split_dst[0], CV_32FC1, 1.0 / (255 * g_std[0]), -1.0 * g_mean[0] / g_std[0]);
  split_dst[1].convertTo(split_dst[1], CV_32FC1, 1.0 / (255 * g_std[1]), -1.0 * g_mean[1] / g_std[1]);
  split_dst[2].convertTo(split_dst[2], CV_32FC1, 1.0 / (255 * g_std[2]), -1.0 * g_mean[2] / g_std[2]);
}

void TvmSsdMxnet::PreProcess(string &jpg_image)  {
  vector<uchar> data(jpg_image.begin(), jpg_image.end());
  cv::Mat image = cv::imdecode(data, CV_LOAD_IMAGE_COLOR);
  PreProcess(image);
}

void TvmSsdMxnet::Predict() {
  // set input
  tvm::runtime::PackedFunc set_input = mod_.GetFunction("set_input");
  set_input("data", input_);

  // DumpDataToFile(".tvm/input.bin", input);

  // do inference
  tvm::runtime::PackedFunc run = mod_.GetFunction("run");
  run();
}

void TvmSsdMxnet::PostProcess(vector<float>         &class_results,
                              vector<float>         &score_results,
                              vector<vector<float>> &bbox_results) {
  // get output
  tvm::runtime::PackedFunc get_output = mod_.GetFunction("get_output");
  tvm::runtime::NDArray class_buf = get_output(0);
  tvm::runtime::NDArray score_buf = get_output(1);
  tvm::runtime::NDArray bbox_buf = get_output(2);

  float *class_pt = reinterpret_cast<float *>(class_buf->data);
  float *score_pt = reinterpret_cast<float *>(score_buf->data);
  float *bbox_pt = reinterpret_cast<float *>(bbox_buf->data);

  for (int i = 0; i < score_buf->shape[1]; i++) {
    if (score_pt[i] < THRESH) break;
    class_results.push_back(class_pt[i]);
    score_results.push_back(score_pt[i]);
    vector<float> tmp;
    tmp.push_back(bbox_pt[i * 4]);
    tmp.push_back(bbox_pt[i * 4 + 1]);
    tmp.push_back(bbox_pt[i * 4 + 2]);
    tmp.push_back(bbox_pt[i * 4 + 3]);
    bbox_results.push_back(tmp);
  }

  return;
}

} // namespace DLRU
