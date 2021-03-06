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
    if [ "$1" = "gpu" ] || [ "$1" = "GPU" ]; then
        cat cmake/config.cmake | sed -e 's/USE_LLVM OFF/USE_LLVM \/usr\/lib\/llvm-5.0\/bin\/llvm-config/g' -e 's/USE_CUDA OFF/USE_CUDA ON/g' > build/config.cmake
    else
        cat cmake/config.cmake | sed -e 's/USE_LLVM OFF/USE_LLVM \/usr\/lib\/llvm-5.0\/bin\/llvm-config/g' > build/config.cmake
    fi
    pushd build
    cmake -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR ..
    make -j8
    make install
    cp -a $TVM_HOME/3rdparty/dlpack/include/dlpack/ $INSTALL_DIR/include
    cp -a $TVM_HOME/3rdparty/dmlc-core/include/dmlc/ $INSTALL_DIR/include
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
    cp -a ../third-party/rapidjson/include/rapidjson $INSTALL_DIR/include
    popd
    popd
}

function build_dlru()
{
    pushd $DLRU_HOME
    mkdir -p build
    pushd build
    cmake -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR ..
    make VERBOSE=1
    make install
    popd
    popd
}

function test_dlru()
{
    pushd $DLRU_HOME/build
    if [ "$1" != "" ]; then
        ctest -V --timeout 10000 -R $1
    else
        ctest -V --timeout 10000
    fi
    popd
}

function clean()
{
    rm -fr $INSTALL_DIR
    rm -fr $DLRU_HOME/build
    rm -fr $TVM_HOME/build
    rm -fr $PISTACHE_HOME/build
}

if [ "$1" = "clean" ]; then
    clean
elif [ "$1" = "test" ]; then
    test_dlru $2
else
    build_tvm $1
    build_pistache
    build_dlru
    cp -a $DLRU_HOME/res $INSTALL_DIR
fi
