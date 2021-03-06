#ifndef __DLRU_UTILS_HPP__
#define __DLRU_UTILS_HPP__

#include <string>
#include <dlpack/dlpack.h>

namespace DLRU {

void DumpDataToFile(const std::string &name, char *data, int size);
void DumpDataToFile(const std::string &name, const std::string &data);
void DumpDataToFile(const std::string &name, DLTensor *data);

}

#endif // __DLRU_UTILS_HPP__
