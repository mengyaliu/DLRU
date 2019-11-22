from flask import Flask
from flask_restful import Resource, Api, reqparse
import werkzeug
import json
import sys
from ssd_mxnet.deploy_ssd import SSD

mode = sys.argv[1]
tmp_dir = sys.argv[2]
app = Flask(__name__)
api = Api(app)
ssd = SSD(mode, tmp_dir)

class BasicTest(Resource):
    def get(self):
        return {'hello': 'world'}
    def post(self):
        parse = reqparse.RequestParser()
        parse.add_argument('file', type=werkzeug.datastructures.FileStorage, location='files')
        args = parse.parse_args()
        tmpFile = args['file']
        tmpFile.save(tmp_dir + "tmp.jpg")
        results = ssd.run(tmp_dir + "tmp.jpg")
        return results

api.add_resource(BasicTest, '/api/v1/object/ssd/jpg')

if __name__ == '__main__':
    app.run(debug=True)
