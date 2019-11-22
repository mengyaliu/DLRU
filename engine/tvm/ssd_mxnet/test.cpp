#include <string>
#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include "TvmSsdMxnet.hpp"

using std::cout;
using std::endl;
using std::string;
using std::vector;

int main(int argc, char *argv[]) {

  if (argc < 4) {
    cout << "Usage: ./test mode model_dir pic" << endl;
    return -1;
  }

  google::InitGoogleLogging(argv[0]);

  string mode(argv[1]);
  string model_dir(argv[2]);
  string lib = model_dir + "deploy_lib.tar.so";
  string graph = model_dir + "deploy_graph.json";
  string params = model_dir + "deploy_param.params";

  // init tvm engine
  DLRU::TvmSsdMxnet engine(mode, lib, graph, params);

  cv::Mat image = cv::imread(argv[3]);

  // preprocess image
  engine.PreProcess(image);

  // do inference
  engine.Predict();

  // post process
  vector<float> class_results, score_results;
  vector<vector<float>> bbox_results;
  engine.PostProcess(class_results, score_results, bbox_results);

  // convert results to json string
  rapidjson::StringBuffer strBuf;
  rapidjson::Writer<rapidjson::StringBuffer> writer(strBuf);
  writer.StartObject();

  for (unsigned int i = 0; i < score_results.size(); i++) {
    writer.Key(std::to_string(i).c_str());
    writer.StartObject();
    writer.Key("score");
    writer.Double(score_results[i]);
    writer.Key("class");
    writer.Double(class_results[i]);
    writer.Key("bbox");
    writer.StartArray();
    vector<float> tmp = bbox_results[i];
    writer.Double(tmp[0]);
    writer.Double(tmp[1]);
    writer.Double(tmp[2]);
    writer.Double(tmp[3]);
    writer.EndArray();
    writer.EndObject();
  }

  writer.EndObject();
  cout << strBuf.GetString() << endl;
  return 0;
}
