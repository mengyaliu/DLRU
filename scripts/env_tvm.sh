#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOP_DIR="${SCRIPT_DIR}/../"
TVM_HOME="${TOP_DIR}/3rdparty/incubator-tvm"
export TVM_HOME
export PYTHONPATH=$TOP_DIR/utils:$TVM_HOME/python:$TVM_HOME/topi/python:$TVM_HOME/nnvm/python:${PYTHONPATH}
export LD_LIBRARY_PATH=$TVM_HOME/build:${LD_LIBRARY_PATH}
