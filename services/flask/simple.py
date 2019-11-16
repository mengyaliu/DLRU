from flask import Flask
from flask_restful import Resource, Api, reqparse
import werkzeug
import json
from ssd_mxnet.deploy_ssd import SSD

app = Flask(__name__)
api = Api(app)
ssd = SSD()

class BasicTest(Resource):
    def get(self):
        return {'hello': 'world'}
    def post(self):
        parse = reqparse.RequestParser()
        parse.add_argument('file', type=werkzeug.datastructures.FileStorage, location='files')
        args = parse.parse_args()
        tmpFile = args['file']
        tmpFile.save(".tvm/tmp.jpg")
        results = ssd.run(".tvm/tmp.jpg")
        return results

api.add_resource(BasicTest, '/api/v1/object/ssd')

if __name__ == '__main__':
    app.run(debug=True)
