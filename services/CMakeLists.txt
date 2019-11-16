cmake_minimum_required(VERSION 3.2.2)

include_directories("${CMAKE_INSTALL_PREFIX}/include")
link_directories("${CMAKE_INSTALL_PREFIX}/lib")

install(FILES flask/simple.py DESTINATION flask)
