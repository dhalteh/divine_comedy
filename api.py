import logging

import flask
from flasgger import Swagger
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from predict import predict

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize the Flask application
application = Flask(__name__)
CORS(application)
swagger = Swagger(application)

def clienterror(error):
    resp = jsonify(error)
    resp.status_code = 400
    return resp


def notfound(error):
    resp = jsonify(error)
    resp.status_code = 404
    return resp


@application.route('/v1/output', methods=['POST'])
def sentiment_classification():
    """Generate text of length N based on provided seed.
        ---
        parameters:
          - name: body
            in: body
            schema:
              id: text
              required:
                - text
                - length
              properties:
                text:
                  type: string
                length:
                    type: integer
            description: the required text for POST method
            required: true
        definitions:
          GeneratedResponse:
          Project:
            properties:
              status:
                type: string
              ml-result:
                type: object
        responses:
          40x:
            description: Client error
          200:
            description: Text Generation Response
            examples:
                          [
{
  "status": "success",
  "sentiment": "1"
},
{
  "status": "error",
  "message": "Exception caught"
},
]
        """
    json_request = request.get_json()
    if not json_request:
        return Response("No json provided.", status=400)
    text = json_request['text']
    length = json_request['length']
    if text is None:
        return Response("No text provided.", status=400)
    else:
        output = predict(input_seed=text, output_length=length)
        return flask.jsonify({"status": "success", "output": output})


if __name__ == '__main__':
    application.run(debug=True, use_reloader=True)