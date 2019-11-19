cmake_minimum_required(VERSION 3.2.2)

include_directories("${CMAKE_INSTALL_PREFIX}/include")
include_directories("${CMAKE_SOURCE_DIR}/utils")
link_directories("${CMAKE_INSTALL_PREFIX}/lib")
link_directories("${CMAKE_BUILD_DIR}/utils")

set(OPENCV_LIBS opencv_core opencv_highgui opencv_imgproc)

set(SRC tvm/ssd_mxnet/TvmSsdMxnet.cpp)

add_library(dlru SHARED ${SRC})

add_executable(test_tvm_ssd_mxnet tvm/ssd_mxnet/test.cpp)
target_link_libraries(test_tvm_ssd_mxnet dlru ${OPENCV_LIBS} tvm_runtime)

install(TARGETS test_tvm_ssd_mxnet DESTINATION bin)
install(TARGETS dlru DESTINATION lib)
install(FILES tvm/ssd_mxnet/compile_ssd.py DESTINATION ssd_mxnet)
install(FILES tvm/ssd_mxnet/deploy_ssd.py DESTINATION ssd_mxnet)
install(FILES tvm/ssd_mxnet/TvmSsdMxnet.hpp DESTINATION include)
