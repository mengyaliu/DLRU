#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DLRU_HOME=$SCRIPT_DIR/..
TVM_HOME=$DLRU_HOME/3rdparty/incubator-tvm

export TVM_HOME DLRU_HOME
export PYTHONPATH=$DLRU_HOME/utils:$TVM_HOME/python:$TVM_HOME/topi/python:$TVM_HOME/nnvm/python:${PYTHONPATH}
export LD_LIBRARY_PATH=$DLRU_HOME/install/lib:${LD_LIBRARY_PATH}
