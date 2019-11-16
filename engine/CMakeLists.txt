cmake_minimum_required(VERSION 3.2.2)

include_directories("${CMAKE_INSTALL_PREFIX}/include")
link_directories("${CMAKE_INSTALL_PREFIX}/lib")

set(OPENCV_LIBS opencv_core opencv_highgui opencv_imgproc)

add_executable(deploy_ssd tvm/ssd_mxnet/deploy_ssd.cpp)
target_link_libraries(deploy_ssd ${OPENCV_LIBS} tvm_runtime)

install(TARGETS deploy_ssd DESTINATION ssd_mxnet)
install(FILES tvm/ssd_mxnet/compile_ssd.py DESTINATION ssd_mxnet)
install(FILES tvm/ssd_mxnet/deploy_ssd.py DESTINATION ssd_mxnet)


