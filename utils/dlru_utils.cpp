#include <fstream>
#define DMLC_USE_GLOG 1
#include <dmlc/io.h>
#include <dmlc/memory_io.h>
#include <tvm/runtime/packed_func.h>

#include "dlru_utils.hpp"

namespace DLRU {

void DumpDataToFile(const std::string &name, char *data, int size) {
  std::fstream fp(name, std::ios::out | std::ios::binary);
  CHECK(!fp.fail()) << "Cannot open " << name;
  fp.write(data, size);
  fp.close();
}

void DumpDataToFile(const std::string &name, const std::string &data) {
  std::fstream fp(name, std::ios::out | std::ios::binary);
  CHECK(!fp.fail()) << "Cannot open " << name;
  fp.write(&data[0], data.length());
}

void DumpDataToFile(const std::string &name, DLTensor *data) {
  std::string bytes;
  dmlc::MemoryStringStream strm(&bytes);
  dmlc::Stream *fp = &strm;
  tvm::runtime::SaveDLTensor(fp, data);
  DumpDataToFile(name, bytes);
}

}
