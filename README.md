# DLRU
Deep Learning Restful interface Utilities

This project will share some example utilities for people who want to develop restful web services providing deep learning algorithm APIs.
For example, an API named "http://localhost:5000/api/v1/object/ssd" can be visited by a POST method with a JPEG file as payload, then JSON format results will be replied.

## Core Features
* Cross-platform(X86 & ARM) supported
* CPU & GPU mode supported
* C++ & Python & Go supported

## Current Status 
* Complete a basic example of flask + mxnet ssd in python.

## TODO
* add another example of restful + onnx ssd in C++

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

* Run and enter interactive mode
```
$ ./docker/run.sh dlru/cpu
```

* Build tvm and export env variables(in container environment now)
```
$ ./scripts/build_tvm.sh
$ source ./scripts/env_tvm.sh
```

* Compile mxnet ssd model
```
$ python3 ./engine/tvm/ssd_mxnet/compile_ssd.py
```

* Run restful service
```
python3 services/flask/simple.py
```

* open another terminal, and test above api
```
curl -v -X POST -H "Content-Type: multipart/form-data" -F "file=@res/street_small.jpg" http://localhost:5000/api/v1/object/ssd
```

then following outputs means the service in container is correct.
```
{
    "3": {
        "class": [
            14.0
        ],
        "score": [
            0.9711700081825256
        ],
        "bbox": [
            22.508594512939453,
            216.10154724121094,
            107.1043701171875,
            362.7880859375
        ]
    },
    "4": {
        "class": [
            14.0
        ],
        "score": [
            0.9478515982627869
        ],
        "bbox": [
            180.46783447265625,
            211.09066772460938,
            255.36398315429688,
            332.4722595214844
        ]
    },
    "2": {
        "class": [
            14.0
        ],
        "score": [
            0.9835368990898132
        ],
        "bbox": [
            354.5447998046875,
            180.3862762451172,
            443.8759765625,
            384.1373291015625
        ]
    },
    "0": {
        "class": [
            1.0
        ],
        "score": [
            0.9996178150177002
        ],
        "bbox": [
            302.73797607421875,
            268.0087890625,
            480.16583251953125,
            395.59930419921875
        ]
    },
    "1": {
        "class": [
            6.0
        ],
        "score": [
            0.9978207349777222
        ],
        "bbox": [
            251.633056640625,
            225.4925994873047,
            383.31536865234375,
            296.90850830078125
        ]
    }
}
```
