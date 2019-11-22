#ifndef __TVMSSDMXNET_HPP__
#define __TVMSSDMXNET_HPP__

#include <vector>
#include <string>
#define DMLC_USE_GLOG 1
#include <dlpack/dlpack.h>
#include <opencv2/opencv.hpp>
#include <tvm/runtime/module.h>

namespace DLRU {

class TvmSsdMxnet {
 public:
  TvmSsdMxnet();
  TvmSsdMxnet(const std::string &mode,
              const std::string &lib,
              const std::string &graph,
              const std::string &params);
  void Init(const std::string &mode,
            const std::string &lib,
            const std::string &graph,
            const std::string &params);
  ~TvmSsdMxnet();
  void PreProcess(cv::Mat &image);
  void PreProcess(std::string &jpg_image);
  void Predict();
  void PostProcess(std::vector<float>              &class_results,
                   std::vector<float>              &score_results,
                   std::vector<std::vector<float>> &bbox_results);
 private:
  float thresh_;
  std::vector<float> mean_;
  std::vector<float> std_;
  DLTensor *input_;
  tvm::runtime::Module mod_;
};

}
#endif // __TVMSSDMXNET_HPP__
