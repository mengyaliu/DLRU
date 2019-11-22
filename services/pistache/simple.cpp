#include <algorithm>

#include <pistache/http.h>
#include <pistache/router.h>
#include <pistache/endpoint.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include "base64.hpp"
#include "dlru_utils.hpp"
#include "TvmSsdMxnet.hpp"

using namespace std;
using namespace Pistache;

class DLEndpoint {
public:
  DLEndpoint(Address addr) : httpEndpoint(std::make_shared<Http::Endpoint>(addr))
  { }

  void init(string mode, string model_dir, size_t thr = 2) {
    auto opts = Http::Endpoint::options()
                  .threads(thr)
                  .maxRequestSize(1024 * 1024 * 20);
    httpEndpoint->init(opts);
    setupRoutes();

    string lib = model_dir + "deploy_lib.tar.so";
    string graph = model_dir + "deploy_graph.json";
    string params = model_dir + "deploy_param.params";

    // init tvm runtime
    runtime_.Init(mode, lib, graph, params);
  }

  void start() {
    httpEndpoint->setHandler(router.handler());
    httpEndpoint->serve();
  }

private:
  void setupRoutes() {
    using namespace Rest;
    Routes::Post(router, "/api/v1/object/ssd/base64", Routes::bind(&DLEndpoint::handleSSD, this));
  }

  void handleSSD(const Rest::Request& request, Http::ResponseWriter response) {
    string jpg_image = DLRU::Base64Decode(request.body());
    // DLRU::DumpDataToFile(".tvm/baset.bin", request.body());
    // DLRU::DumpDataToFile(".tvm/jpg.bin", jpg_image);
    runtime_.PreProcess(jpg_image);
    runtime_.Predict();
    vector<float> class_results, score_results;
    vector<vector<float>> bbox_results;
    runtime_.PostProcess(class_results, score_results, bbox_results);

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
    response.send(Http::Code::Ok, strBuf.GetString());
  }

  std::shared_ptr<Http::Endpoint> httpEndpoint;
  Rest::Router router;
  DLRU::TvmSsdMxnet runtime_;
};

int main(int argc, char *argv[]) {

  int thr = 4;
  Port port(8000);

  if (argc < 3) {
    cout << "Usage: " << argv[0] << " <mode> <model_dir> [port] [thread_num]";
    exit(0);
  }

  string mode(argv[1]);
  string model_dir(argv[2]);
  if (argc >= 4) {
    port = std::stol(argv[3]);
    if (argc == 5)
      thr = std::stol(argv[4]);
  }

  google::InitGoogleLogging(argv[0]);

  Address addr(Ipv4::any(), port);

  cout << "Cores = " << hardware_concurrency() << endl;
  cout << "Using " << thr << " threads" << endl;

  DLEndpoint engine(addr);

  engine.init(mode, model_dir, thr);
  engine.start();
}
