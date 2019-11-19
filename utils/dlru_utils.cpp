#include <fstream>
#include <dmlc/io.h>
#include <dmlc/memory_io.h>
#include <tvm/runtime/packed_func.h>

#include "dlru_utils.hpp"

namespace DLRU {

void DumpDataToFile(const std::string &name, char *data, int size) {
  std::fstream fp(name, std::ios::out | std::ios::binary);
  fp.write(data, size);
  fp.close();
}

void DumpDataToFile(const std::string &file_name, const std::string &data) {
  std::ofstream fs(file_name, std::ios::out | std::ios::binary);
  CHECK(!fs.fail()) << "Cannot open " << file_name;
  fs.write(&data[0], data.length());
}

void DumpDataToFile(const std::string &name, DLTensor *data) {
  std::string bytes;
  dmlc::MemoryStringStream strm(&bytes);
  dmlc::Stream *fp = &strm;
  tvm::runtime::SaveDLTensor(fp, data);
  DumpDataToFile(name, bytes);
}

}
