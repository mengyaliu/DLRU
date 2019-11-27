from flask import Flask
from flask_restful import Resource, Api, reqparse
import werkzeug
import json
import sys
from ssd_mxnet.deploy_ssd import SSD
from yolo_darknet.deploy_yolo import YOLO

mode = sys.argv[1]
tmp_dir = sys.argv[2]
app = Flask(__name__)
api = Api(app)

ssd_lib = tmp_dir + "/ssd/deploy_lib.tar"
ssd_graph = tmp_dir + "/ssd/deploy_graph.json"
ssd_param = tmp_dir + "/ssd/deploy_param.params"
ssd = SSD(mode, ssd_lib, ssd_graph, ssd_param)

yolo_lib = tmp_dir + "/yolo/deploy_lib.tar"
yolo_graph = tmp_dir + "/yolo/deploy_graph.json"
yolo_param = tmp_dir + "/yolo/deploy_param.params"
yolo_data = tmp_dir + "/yolo/data.npy"
yolo = YOLO(mode, 'yolov3', yolo_lib, yolo_graph, yolo_param, yolo_data)

class YOLOTest(Resource):
    def get(self):
        return {'algorithm': 'yolov3'}
    def post(self):
        parse = reqparse.RequestParser()
        parse.add_argument('file', type=werkzeug.datastructures.FileStorage, location='files')
        args = parse.parse_args()
        tmpFile = args['file']
        tmpFile.save(tmp_dir + "/yolo/tmp.jpg")
        results = yolo.run(tmp_dir + "/yolo/tmp.jpg")
        return results

class SSDTest(Resource):
    def get(self):
        return {'algorithm': 'ssd'}
    def post(self):
        parse = reqparse.RequestParser()
        parse.add_argument('file', type=werkzeug.datastructures.FileStorage, location='files')
        args = parse.parse_args()
        tmpFile = args['file']
        tmpFile.save(tmp_dir + "/ssd/tmp.jpg")
        results = ssd.run(tmp_dir + "/ssd/tmp.jpg")
        return results

api.add_resource(SSDTest, '/api/v1/object/ssd/jpg')
api.add_resource(YOLOTest, '/api/v1/object/yolo/jpg')

if __name__ == '__main__':
    app.run(debug=True)
