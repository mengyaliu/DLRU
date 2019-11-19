cmake_minimum_required(VERSION 3.2.2)

include_directories("${CMAKE_INSTALL_PREFIX}/include")

add_library(dlru_utils STATIC dlru_utils.cpp)
target_link_libraries(dlru_utils tvm_runtime)

install(TARGETS dlru_utils DESTINATION lib)
install(FILES timerecorder.py DESTINATION utils)
install(FILES base64.hpp DESTINATION include)
install(FILES dlru_utils.hpp DESTINATION include)
