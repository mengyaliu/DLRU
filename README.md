# DLRU
Deep Learning Restful interface Utilities

This project will share some example utilities for people who want to develop restful web services providing deep learning algorithm APIs.
For example, an API named "http://localhost:5000/api/v1/object/ssd" can be visited by a POST method with a JPEG file as payload, then JSON format results will be replied.

## Core Features
* Cross-platform(X86 & ARM) supported
* CPU & GPU mode supported
* C++ & Python & Go supported

## Current Status 
* Complete an example of flask + cpu tvm mxnet ssd in python.
* Complete an example of pistach + cpu tvm mxnet ssd in c++.

## TODO
* add GPU tvm support

## Get Started

### Prepare develop environment and checkout codes

* Basic requirement

First of all, prepare a proper develop environment supporting git && docker, Ubuntu/CentOS/Win10/MacOS will be all OK.

Here we give example steps of Ubuntu16.04.

* Fetch source codes
```
$ git clone --recursive https://github.com/mengyaliu/DLRU.git
```

### Build for CPU mode

* Build docker images
```
$ ./docker/build.sh cpu
```

* Run and enter interactive mode
```
$ ./docker/run.sh dlru/cpu
```

* After entering interactive mode, build all and export env variables
```
$ source ./scripts/env.sh
$ ./scripts/build.sh
$ ./scripts/build.sh test cpu
```

### Build for GPU mode

* Build docker images
```
$ ./docker/build.sh gpu
```

* Run and enter interactive mode
```
$ ./docker/run.sh dlru/gpu
```

* After entering interactive mode, build all and export env variables
```
$ source ./scripts/env.sh
$ ./scripts/build.sh gpu
$ ./scripts/build.sh test gpu
```

### Try pistache c++ restful service example, cpu mode

* Start pistache service
```
./install/bin/simple_pistache cpu .model/cpu/
```

* open another terminal, and test above api
```
curl -d "@res/street_small_base64.txt"  http://localhost:8000/api/v1/object/ssd/base64
```

then following outputs means the service in container is correct.
```
{"0":{"score":0.999661922454834,"class":1.0,"bbox":[302.7614440917969,267.5565185546875,479.8902282714844,395.59326171875]},"1":{"score":0.9974727034568787,"class":6.0,"bbox":[252.70074462890626,224.68527221679688,380.7926025390625,296.6217346191406]},"2":{"score":0.9877620935440064,"class":14.0,"bbox":[355.84759521484377,180.6073455810547,443.7603759765625,384.37005615234377]},"3":{"score":0.9850497841835022,"class":14.0,"bbox":[178.17184448242188,206.58700561523438,259.6250915527344,330.6838684082031]},"4":{"score":0.9553403854370117,"class":14.0,"bbox":[23.591705322265626,218.72982788085938,103.66038513183594,361.9799499511719]}}
```

### Try flask python restful service example, cpu mode

* Start service
```
python3 ./install/flask/simple.py cpu .model/cpu/
```

* open another terminal, and test above api
```
curl -X POST -F "file=@res/dog.jpg" http://localhost:5000/api/v1/object/yolo/jpg
```

then following outputs means the service in container is correct.
```
{
    "2": {
        "bbox": [
            164,
            562,
            112,
            444
        ],
        "class": "bicycle",
        "score": "0.99416536"
    },
    "0": {
        "bbox": [
            128,
            314,
            224,
            536
        ],
        "class": "dog",
        "score": "0.98993456"
    },
    "1": {
        "bbox": [
            473,
            688,
            85,
            170
        ],
        "class": "truck",
        "score": "0.92719495"
    }
}
```

### Try flask python restful service example, gpu mode

* Start service
```
python3 ./install/flask/simple.py gpu .model/gpu/
```

* open another terminal, and test above api
```
curl -X POST -F "file=@res/street_small.jpg" http://localhost:5000/api/v1/object/ssd/jpg
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
