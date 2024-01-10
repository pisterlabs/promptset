from collections import Counter
from common.entities import StockDirection, Datasets, AnswerDataPoint
from common.data_loading import load_data_splits
from tqdm import tqdm
from common.utils import llama2_token_length
from dotenv import load_dotenv
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from random import seed, shuffle
import os 
from openai import OpenAI
import huggingface_hub
import json
from datasets import load_dataset

load_dotenv()

SEED = os.environ['SEED']
seed(SEED)

HUGGINGFACE_TOKEN = os.environ['HUGGINGFACE_TOKEN']
huggingface_hub.login(token=HUGGINGFACE_TOKEN)

openai_client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
dataset_name = 'michelcarroll/llama2-earnings-stock-prediction-fine-tune-binary'

NUM_EXAMPLES_TO_EVALUATE = 1000

split_name = 'test'

class OpenAIModel:

    def __init__(self, model: str) -> None:
        self._model = model
        
    def classify(self, text: str) -> StockDirection: 
        response = self._openai_request(text=text)

        if StockDirection.Up.value in response:
            return StockDirection.Up
        elif StockDirection.Down.value in response:
            return StockDirection.Down
        else:
            raise Exception(f"Response did not contain one of the classes: {response}")

    def _openai_request(self, text: str) -> str:

        chat_completion = openai_client.chat.completions.create(
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": f"You are a binary classifier with expert financial analyst knowledge, predicting which direction the stock price will go following this answer from the Q/A section of an earnings call. Output either UP if you predict the stock will go up, or DOWN if you predict it will go down. You must absolutely make a prediction – don't answer with N/A.",
                },
                {
                    "role": "user",
                    "content": f"The answer from the earnings transcript is: {text}",
                }
            ],
            functions = [{
                "name": "predict",
                "description": "Label the correct class",
                "parameters": {
                    "type": "object",
                    "properties": {
                        # 'reasoning': {
                        #     "type": "string"
                        # },
                        'prediction': {
                            "type": "string",
                            "enum": ["UP", "DOWN"]
                        },
                    },
                    "required": [ "prediction"]
                }
            }],
            function_call={'name': 'predict'},
            model=self._model,
            timeout=15
        )
        arguments = json.loads(chat_completion.choices[0].message.function_call.arguments)
        # print(arguments['reasoning'])
        return arguments['prediction']


def filter_answer(answer: str, token_length_low_threshold: int = 20, token_length_high_threshold: int = 1000) -> bool:
    text_token_length = llama2_token_length(answer)
    return text_token_length >= token_length_low_threshold and text_token_length <= token_length_high_threshold

def evaluate(label: str, llm_model, datapoints):
    predictions: list[StockDirection] = []
    true_labels: list[StockDirection] = []

    for datapoint in tqdm(datapoints, desc="Evaluating"):
        try:
            result = llm_model.classify(text=datapoint['completion'])
        except Exception as e:
            print("ERROR", e.args[0])
            continue 
        if result:
            predictions.append(result.value)
            true_labels.append(datapoint['label'])

    print("Prediction Counts: ", Counter(predictions))
            
    print("="*10)
    print("Results for ", label)
    print("="*10)
    print("N of ", len(datapoints))
    print("Accuracy Score: ", accuracy_score(y_true=true_labels, y_pred=predictions))
    print("F1 Score: ", f1_score(y_true=true_labels, y_pred=predictions, pos_label='UP'))
    print("Confusion Matrix")
    print(confusion_matrix(y_true=true_labels, y_pred=predictions, labels=["UP", "DOWN"]))


answer_datapoints = load_dataset(dataset_name, split=f"{split_name}[0:{NUM_EXAMPLES_TO_EVALUATE}]")

evaluate(
    label="GPT 4 with CoT", 
    llm_model = OpenAIModel(model='gpt-4'), 
    datapoints=answer_datapoints
)