#ifndef __TVMSSDMXNET_HPP__
#define __TVMSSDMXNET_HPP__

#include <vector>
#include <string>
#include <dlpack/dlpack.h>
#include <opencv2/opencv.hpp>
#include <tvm/runtime/module.h>

namespace DLRU {

class TvmSsdMxnet {
 public:
  float thresh_;
  std::vector<float> mean_;
  std::vector<float> std_;
  DLTensor *input_;
  tvm::runtime::Module mod_;

  TvmSsdMxnet(const std::string &lib,
              const std::string &graph,
              const std::string &params);
  ~TvmSsdMxnet();
  void PreProcess(cv::Mat &image);
  void Predict();
  void PostProcess(std::vector<float>              &class_results,
                   std::vector<float>              &score_results,
                   std::vector<std::vector<float>> &bbox_results);
};

}
#endif // __TVMSSDMXNET_HPP__
