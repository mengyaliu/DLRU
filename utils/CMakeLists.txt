cmake_minimum_required(VERSION 3.2.2)

set(SRC ${CMAKE_CURRENT_SOURCE_DIR}/dlru_utils.cpp)

include_directories("${CMAKE_INSTALL_PREFIX}/include")
link_directories("${CMAKE_INSTALL_PREFIX}/lib")

add_library(dlru_utils STATIC ${SRC})
target_link_libraries(dlru_utils tvm_runtime)

install(FILES timerecorder.py DESTINATION utils)
install(FILES base64.hpp DESTINATION include)
