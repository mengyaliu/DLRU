#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DLRU_HOME=$SCRIPT_DIR/../
INSTALL_DIR=$DLRU_HOME/install
TVM_HOME=$DLRU_HOME/3rdparty/incubator-tvm
PISTACHE_HOME=$DLRU_HOME/3rdparty/pistache

mkdir -p $INSTALL_DIR

function build_tvm()
{
    pushd $TVM_HOME
    mkdir -p build
    cat cmake/config.cmake | sed -e 's/USE_LLVM OFF/USE_LLVM \/usr\/lib\/llvm-5.0\/bin\/llvm-config/g' > build/config.cmake
    pushd build
    cmake -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR ..
    make -j8
    make install
    popd
    popd
}

function build_pistache()
{
    pushd $PISTACHE_HOME
    mkdir -p build
    pushd build
    cmake -G "Unix Makefiles" \
        -DCMAKE_BUILD_TYPE=Release \
        -DPISTACHE_BUILD_EXAMPLES=true \
        -DPISTACHE_BUILD_TESTS=true \
        -DPISTACHE_BUILD_DOCS=false \
        -DPISTACHE_USE_SSL=true \
        -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
        ../
    make -j8
    make install
}

function clean()
{
  rm -fr $INSTALL_DIR
  rm -fr $TVM_HOME/build
  rm -fr $PISTACHE_HOME/build
}

if [ "$1" = "clean" ]; then
    clean
else
    build_tvm
    build_pistache
fi