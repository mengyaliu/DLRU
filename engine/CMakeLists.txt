cmake_minimum_required(VERSION 3.2.2)

include_directories("${CMAKE_SOURCE_DIR}/utils")
include_directories("${CMAKE_INSTALL_PREFIX}/include")
link_directories("${CMAKE_INSTALL_PREFIX}/lib")

find_package(PkgConfig REQUIRED)
pkg_search_module(OPENCV REQUIRED opencv)

set(SRC tvm/ssd_mxnet/TvmSsdMxnet.cpp)

add_library(dlru SHARED ${SRC})
target_link_libraries(dlru dlru_utils ${OPENCV_LIBRARIES})

add_executable(test_tvm_ssd_mxnet tvm/ssd_mxnet/test.cpp)
target_link_libraries(test_tvm_ssd_mxnet dlru)

install(TARGETS test_tvm_ssd_mxnet DESTINATION bin)
install(TARGETS dlru DESTINATION lib)
install(FILES tvm/ssd_mxnet/compile_ssd.py DESTINATION ssd_mxnet)
install(FILES tvm/ssd_mxnet/deploy_ssd.py DESTINATION ssd_mxnet)
install(FILES tvm/ssd_mxnet/TvmSsdMxnet.hpp DESTINATION include)
install(FILES tvm/yolo_darknet/compile_yolo.py DESTINATION yolo_darknet)
install(FILES tvm/yolo_darknet/deploy_yolo.py DESTINATION yolo_darknet)

set(RES_DIR ${CMAKE_SOURCE_DIR}/res)
set(TMP_DIR ${CMAKE_SOURCE_DIR}/.model/)
set(ENGINE_TVM_HOME ${CMAKE_SOURCE_DIR}/engine/tvm)

include(CTest)
add_test(test_tvm_ssd_mxnet_compile python3 ${ENGINE_TVM_HOME}/ssd_mxnet/compile_ssd.py cpu ${TMP_DIR})
add_test(test_tvm_ssd_mxnet_deploy_py python3 ${ENGINE_TVM_HOME}/ssd_mxnet/deploy_ssd.py cpu ${TMP_DIR} ${RES_DIR}/street_small.jpg)
add_test(test_tvm_ssd_mxnet_deploy_cpp test_tvm_ssd_mxnet cpu ${TMP_DIR} ${RES_DIR}/street_small.jpg)
add_test(test_tvm_yolo_darknet_compile python3 ${ENGINE_TVM_HOME}/yolo_darknet/compile_yolo.py cpu yolov3 ${TMP_DIR})
add_test(test_tvm_yolo_darknet_deploy_py python3 ${ENGINE_TVM_HOME}/yolo_darknet/deploy_yolo.py cpu yolov3 ${TMP_DIR} ${RES_DIR}/dog.jpg)
