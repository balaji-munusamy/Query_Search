from flask import Flask, request, jsonify
import json
from shutil import copyfile
import os, ast
from os.path import basename
import logging
import query_search

approot = os.path.dirname(os.path.abspath(__file__))+"\\"

app = Flask(__name__)

class QueryException(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        rv['status'] = self.status_code
        return rv


@app.errorhandler(QueryException)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response

@app.route('/query_interface/healthstatus', methods = ['GET'])
def getHealthstatus():
    return "service is up"

#@app.route('/v1/convertFileToJSON', methods = ['GET'])
#def convertFileToJSON():
#    config_file='config.txt'
#    configData=ast.literal_eval(open(approot+config_file,'r').read())
#    return json.dumps(configData)

	
@app.route('/query_interface/respond_to_query', methods = ['POST'])
def respond_to_query():
    req = request.json; 
    response = {}
    status = 1
    output_resp = ""
    try:
        input_query = req['query']
        output_resp = query_search.query_interface(input_query)
        
    except Exception as ae:
           logging.exception(ae)
           status=500
           err_msg = ""
           if hasattr(ae, 'message'):
               err_msg = ae.message
           else:
               err_msg = str(ae)
           raise QueryException(err_msg, status_code=status)
    response['response'] = output_resp
    response['status'] = status
    response=json.dumps(response); print(response)
    return response

#@app.route('/v1/trin', methods = ['POST'])
#def train():
#    req = request.json; 
#    response = {}
#    status = 1
#    train_resp = ""
#    try:
#        inputfile_rest = req['trainFile']
#        filename = basename(inputfile_rest)
#        inputfile = approot+"supervised"+"/"+"input"+"/"+filename
#        file_exist = os.path.exists(inputfile)
#        if not file_exist:
#            os.rename(inputfile_rest, inputfile)
#        train_resp = ai_train.train_pipeline(inputfile, inputdir='', learn_dict={}, config_dict={})
#    except Exception as ae:
#           logging.exception(ae)
#           status=500
#           err_msg = ""
#           if hasattr(ae, 'message'):
#               err_msg = ae.message
#           else:
#               err_msg = str(ae)
#           raise QueryException(err_msg, status_code=status)
#    response['ai_response'] = train_resp
#    response['ai_status'] = status
#    response=json.dumps(response); print(response)
#    return response	
#
#@app.route('/v1/exception_structure', methods = ['POST'])
#def parseFile():
#    file = request.files['file']; print('input-->', file)
#    filename = file.filename
#    file.save(approot+'/input/'+filename)
#    response = ''
#    try:
#        resValues = information_retrieval.processfile(filename)
#        response = {'status':1,'response':resValues}
#    except ValueError as ve:
#           print(ve) 
#           response = {'status':0}
#    
#    response=json.dumps(response)
#    print('output-->', response)
#    return response
	
if __name__ == "__main__":
    app.run(host='0.0.0.0',port=6001,debug=True)
