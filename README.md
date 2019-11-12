# DLRU
Deep Learning Restful interface Utilities

This project will share some example utilities for people who want to create restful web services providing deep learning algorithm calls.
For example, an API named "http://localhost:8080/api/v1/face/detect/jpg" can be visited by a POST method with a JPEG file as payload, then JSON format results will be replied.

## Core Features
* Cross-platform(X86 & ARM) supported
* CPU & GPU mode supported
* C++ & Python supported

## Get Started

* Basic requirement

First of all, prepare a proper develop environment supporting git && docker, Ubuntu/CentOS/Win10/MacOS will be all OK.

Here we give example steps of Ubuntu16.04.

* Fetch source codes
```
$ git clone https://github.com/mengyaliu/DLRU.git
$ cd DLRU
$ git submodule update --init --recursive
```

* Build docker images
```
$ ./docker/build.sh cpu
```

* Run container
```
$ ./docker/run.sh dlru/cpu
```

* Build tvm and export env variables
```
$ ./scripts/build_tvm.sh
$ source ./scripts/env_tvm.sh
```

* Compile mxnet ssd model
```
$ python3 ./engine/tvm/ssd_mxnet/compile_ssd.py
```

* Run inference
```
python3 ./engine/tvm/ssd_mxnet/deploy_ssd.py res/street_small.jpg
```

then following outputs means the script is executed correctly.
```
Result 0 :  [1.] ; [0.9996178] ; [302.73798 268.0088  480.16583 395.5993 ]
Result 1 :  [6.] ; [0.99782073] ; [251.63306 225.4926  383.31537 296.9085 ]
Result 2 :  [14.] ; [0.9835369] ; [354.5448  180.38628 443.87598 384.13733]
Result 3 :  [14.] ; [0.97117] ; [ 22.508595 216.10155  107.10437  362.7881  ]
Result 4 :  [14.] ; [0.9478516] ; [180.46783 211.09067 255.36398 332.47226]
time: <inference>: 586.237549 ms
```
