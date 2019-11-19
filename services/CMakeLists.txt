cmake_minimum_required(VERSION 3.2.2)

include_directories("${CMAKE_SOURCE_DIR}/utils")
include_directories("${CMAKE_SOURCE_DIR}/engine/tvm/ssd_mxnet")
include_directories("${CMAKE_INSTALL_PREFIX}/include")
link_directories("${CMAKE_INSTALL_PREFIX}/lib")

add_executable(simple_pistache pistache/simple.cpp)
target_link_libraries(simple_pistache pistache dlru)

install(FILES flask/simple.py DESTINATION flask)
install(TARGETS simple_pistache DESTINATION bin)
