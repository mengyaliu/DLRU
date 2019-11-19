cmake_minimum_required(VERSION 3.2.2)
project(DLRU)

ENABLE_LANGUAGE(ASM)

message(STATUS "CMAKE_SYSTEM_NAME: " ${CMAKE_SYSTEM_NAME})
message(STATUS "CROSS_COMPILE: " ${CROSS_COMPILE})

macro(use_cxx11)
  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endmacro(use_cxx11)

macro(use_cxx14)
  set(CMAKE_CXX_STANDARD 14)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endmacro(use_cxx14)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -O2")

use_cxx14()

add_subdirectory(utils)
add_subdirectory(engine)
add_subdirectory(services)
