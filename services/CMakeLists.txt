cmake_minimum_required(VERSION 3.2.2)

include_directories("${CMAKE_INSTALL_PREFIX}/include")
include_directories("${CMAKE_SOURCE_DIR}/utils")
include_directories("${CMAKE_SOURCE_DIR}/engine/tvm/ssd_mxnet")
link_directories("${CMAKE_INSTALL_PREFIX}/lib")

set(OPENCV_LIBS opencv_core opencv_highgui opencv_imgproc)

add_executable(simple_pistache pistache/simple.cpp)
target_link_libraries(simple_pistache pistache dlru_utils dlru tvm_runtime ${OPENCV_LIBS})

install(FILES flask/simple.py DESTINATION flask)
install(TARGETS simple_pistache DESTINATION bin)
