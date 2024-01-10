from datetime import datetime
import openai
from helper import prompt, container
from flask import Flask, jsonify, request,json
#from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import logging
from opencensus.ext.azure.log_exporter import AzureLogHandler
#import pandas as pd


openai.api_key = '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'


def gpt_qa(question):
    query = question
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt+ query,
        temperature=0,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["query"]
    )
    tokens = response['choices'][0]['text'].split()
    print(tokens)
    try:
        ind = tokens.index('answer:')
        return ' '.join(tokens[ind+1:])
    except:
        return ' '.join(tokens)


app = Flask(__name__)


@app.route("/", methods=['POST'])
def dummy_api():
    jsondata = request.get_json()
    print(jsondata)
    question = jsondata['question']
    answer = gpt_qa(question)
    result = {'answer': answer}
    #logging
    try:
        logger = logging.getLogger(__name__)
        # TODO: replace the all-zero GUID with your instrumentation key.
        logger.addHandler(AzureLogHandler(
            connection_string='InstrumentationKey=669a9966-eaa5-4419-9685-2450f8dc7c6d')
        )
        logger.info('question: '+question+' ans: '+ answer)
        logger.warning('question: '+question+' ans: '+ answer)
        container.create_item(body={"id":str(datetime.now()),"messageFrom":question})
    except:
        print('error in logging')
    return json.dumps(result), 200


if __name__ == "__main__":
    app.run(debug=True, port=5001)

