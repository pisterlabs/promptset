import openml
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import os 
from ast import literal_eval

_ = load_dotenv(find_dotenv())  # read local .env file
client = OpenAI()
OpenAI.api_key = os.getenv("OPENAI_API_KEY")


# model = "gpt-3.5-turbo-1106" or "gpt-4-1106-preview"
def get_completion(prompt, model="gpt-3.5-turbo-1106"):
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=0
    )
    return response.choices[0].message.content

# transformers version v4.36.1
task_choices = ["AutoModelForCausalLM",
                "AutoModelForMaskedLM",
                "AutoModelForMaskGeneration",
                "AutoModelForSeq2SeqLM",
                "AutoModelForSequenceClassification",
                "AutoModelForMultipleChoice",
                "AutoModelForNextSentencePrediction",
                "AutoModelForTokenClassification",
                "AutoModelForQuestionAnswering",
                "AutoModelForTextEncoding",
                "AutoModelForDepthEstimation",
                "AutoModelForlmageClassification",
                "AutoModelForVideoClassification",
                "AutoModelForMaskedImageModeling",
                "AutoModelForObjectDetection",
                "AutoModelForlmageSegmentation",
                "AutoModelForImageTolmage",
                "AutoModelForSemanticSegmentation",
                "AutoModelForlnstanceSegmentation",
                "AutoModelForUniversalSegmentation",
                "AutoModelForZeroShotlmageClassification",
                "AutoModelForZeroShotObjectDetection",
                "AutoModelForAudioClassification",
                "AutoModelForAudioFrameClassification",
                "AutoModelForCTC",
                "AutoModelForSpeechSeq2Seq",
                "AutoModelForAudioXVector",
                "AutoModelForTextToSpectrogram",
                "AutoModelForTextToWaveform",
                "AutoModelForTableQuestionAnswering",
                "AutoModelForDocumentQuestionAnswering",
                "AutoModelForVisualQuestionAnswering",
                "AutoModelForVision2Seq",
                ]


def openml_task_inference(dataset_name):
    dataset = openml.datasets.get_dataset(dataset_name)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe",
        target=dataset.default_target_attribute,
    )
    X_head = X.head()
    y_head = y.head()
    # print("X_head\n", X_head)
    # print("y_head\n", y_head)    
    # TASK INFERENCE
    taskInference_prompt = f"""
        Your task is to infer the task based on the training data, X_head is the features and y_head is the labels. 
        Both the X_head and y_head are enclosed in the triple backticks.
        you should follow these steps when you infer the task:
        1. Load the X_head and y_head data.
        2. Infer the task based on the X_head and y_head data.
        3. Return the task.
 
        At the end, please return the task.
        
        X_head: ```{X_head}```
        y_head: ```{y_head}```
        
    """
    taskInference_response = get_completion(taskInference_prompt)
    print(taskInference_response)
    
    # MODEL SELECT 
    model_select_prompt = f"""
    Your task is to identify the most suitable model for the following task, The task is enclosed in triple backticks.
    DO not give me explanation information. Only Output a list of the models after you selected and compared. eg. ['microsoft/restnet-50', 'gpt-3.5-turbo-1106', 'gpt-4-1106-preview']. 
    If there are no models suitable, output an empty list [].
    
    task: ```{taskInference_response}```
    """
    #  After selecting the most appropriate model based on your knowledge, compare it with the models in `{models}`. If the model you selected is not in `{models}`, explain why it is more suitable than the ones listed.
    #  Output your findings in JSON format with the following keys: chosen_model, compared_models, query, reason for choice.

    model_select_response = get_completion(model_select_prompt)
    print("model_select_response:\n ", model_select_response)
    model_selected_list = literal_eval(model_select_response)
    most_suitable_model = model_selected_list[0]
    print("most_suitable_model:\n ", most_suitable_model)
    
    
    
    # TRAINER which use Hugging Face Model and Trainer
    hf_model_trainer_prompt = f"""
        Your task is to generate training code snippet for the TASK with the MODEL give you. The TASK and MODEL is enclosed in triple backticks.
        You should follow these steps when you write the code snippet:
        1. Import the necessary libraries and modules,such as openml, sklearn, datasets, transformers etc.
        2. Use 'openml.datasets.get_dataset' function to load the DATASET_NAME I gave you. The DATASET_NAME is enclosed in the triple backticks. for example,get_dataset('CIFAR_10')
        At the end, please return the Trainer code snippet with some usage code snippet.
        3. Use 'get_data' function to get the data from the dataset. dataset_format="dataframe",target=dataset.default_target_attribute. Use X and y to store the data.
        4. Split the data into training and testing sets.
        5. Convert the data to a CSV file for easy reading by the Hugging Face datasets library. You should follow these steps:
            5.1 Create a pandas DataFrame train_df use the X_train, append a column to the DataFrame with the column name 'target' and the values of y_train.
            5.2 Create a pandas DataFrame test_df use the X_test, append a column to the DataFrame with the column name 'target' and the values of y_test.
            5.3 Use to_csv function to generate the csv file.
            5.4 Load the csv file with the load_dataset function.
        6. Initialize the MODEL. Be sure to use the most suitable model based on the MODEL. If the task related to text, use Tokenizer to tokenize the text. If the task related to image classfication, do not use tokenizer but use AutoModelForImageClassification to initialize the model.
        8. Train the model on the train dataset.
        9. Make predictions on the testing set.
        10. Evaluate the model.
        

        MODEL: ```{most_suitable_model}```
        TASK: ```{taskInference_response}```
        DATASET_NAME: ```{dataset_name}```
        
    """
    hf_model_trainer_response = get_completion(hf_model_trainer_prompt, model="gpt-4-1106-preview")
    print("hf_model_trainer_response", hf_model_trainer_response)
    
    with open(f'./generated_scripts/{most_suitable_model}_hf.py', 'w') as f:
        f.write(hf_model_trainer_response)
    
    # TRAINER which not use Hugging Face Model and Trainer
    trainer_prompt = f"""
        Your task is to generate training code snippet for the task with the model give you. The task and model is enclosed in triple backticks.
        You should follow these steps when you write the code snippet:
        1. Import the necessary libraries and modules,such as openml, sklearn, datasets, transformers, Trainer etc.
        2. Use 'openml.datasets.get_dataset' function to load the dataset using the dataset name I gave you. The dataset name is enclosed in the triple backticks.
        At the end, please return the Trainer code snippet with some usage code snippet.
        3. Use 'get_data' function to get the data from the dataset. dataset_format="dataframe",target=dataset.default_target_attribute.
        4. Split the data into training and testing sets.
        5. Initialize the model.
        6. Train the model.
        7. Make predictions on the testing set.
        8. Evaluate the model.
        

        model: ```{most_suitable_model}```
        task: ```{taskInference_response}```
        dataset: ```{dataset_name}```
        
    """
    trainer_response = get_completion(trainer_prompt, model="gpt-4-1106-preview")
    print(trainer_response)
    with open(f'./generated_scripts/{most_suitable_model}.py', 'w') as f:
        f.write(trainer_response)

if __name__ == "__main__":
    # openml_task_inference('CIFAR_10')
    
    openml_task_inference('CIFAR_10')
