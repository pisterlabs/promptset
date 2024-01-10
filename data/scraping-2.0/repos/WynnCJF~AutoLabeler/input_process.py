import os
from dotenv import load_dotenv
from aiconfig import AIConfigRuntime, InferenceOptions, CallbackManager
from aiconfig import Prompt
import pandas as pd
import asyncio
import openai
import google.generativeai as palm
from collections import Counter
from statistics import mode
import streamlit as st

load_dotenv()

openai.api_key = os.getenv("openai_api_key")
palm.configure(api_key=os.getenv("palm_api_key"))

BATCH_NUM = 10

class Labeler:
    def __init__(self, task, desc, task_desc, file_path):
        """
        Constructor creates aiconfig.
        
        Inputs:
            task: string - labelling task description
            desc: string - description of input data
            filepath: string - path to csv file containing training examples
        """
        train_df = pd.read_csv(file_path)
        classes = train_df["label"].unique()

        params = self.__getTrainParams(train_df, classes, task, desc, task_desc)

        # create model and add training parameters
        self.aiconfig = AIConfigRuntime.create(
            "labeler_config",
            "data labeler config",
            metadata={"parameters": params}
        )
        
    def create_labeler(self, name, model_type="gpt-3.5-turbo", model_settings=None):
        """
        Create a lastmileAI model and prompt in aiconfig to be used for prediction.
        
        Inputs:
            name: string - name of labeler
            model_type: string - model of LLM used
            model_settings: python dictionary (contents depend on model used)
        """
        model_name = model_type

        if model_settings == None:
            model_settings = {"top_p": 1, "model": model_type, "temperature": 1, "remember_chat_context": False,
                              "system_prompt": "You will only output a csv. No text explanation unless specified in the prompt."}

        self.aiconfig.add_model(model_name, model_settings)
        label_inputs = Prompt(
            name=name,
            input="""I want you to help with labeling data for the task of {{ task_name }}. Here's a brief summary of the task: {{ task_description }}.\nThe feature input of the task is {{ input_description }}. The output is a {{ class_num }}-class classification result. The output classes are: {{ output_classes }}.\nHere are several examples: {{ examples }}
            Please act as a data labeler, and label the following data: {{ real_inputs }}. There are {{ num_predictions }} inputs and {{ num_prediction}} outputs expected. Return only the predicted outputs separated by a comma.""",
            metadata={"model": {"name": model_type,
                                "settings": {"model": model_type}}}
        )

        self.aiconfig.add_prompt(name, label_inputs)

    def saveAIConfig(self, fname, path="./"):
        """
        Saves AIConfig to a JSON file.
        
        Inputs:
            fname: string - name of file (must end with .json)
            path: directory to save file to
        """
        self.aiconfig.save(path+fname, include_outputs=False)


    async def predict(self, path, labeler, calculate_mode=True, default_mode=""):
        """
        Generate predictions to text inputs.
        
        Inputs:
            path: string - file path to csv file with text inputs (one column, with header)
            labeler: string or list[string] - names of labelers to be used for prediction
            mode: bool - If true and labeler contains 3 elements or more, mode is added to output dataframe

        Returns pandas dataframe with original input and predictions.
        """
        df = pd.read_csv(path)
        total_samples = df.shape[0]
        print("Receiving " + str(total_samples) + " entries in total\n")
        
        if not isinstance(labeler, list):
            labeler = [labeler]
        
        run_iterations = total_samples // BATCH_NUM
        result_df = pd.DataFrame(columns=['Input'] + labeler + ['Mode'])
        
        my_bar = st.progress(0.0)
        for i in range(run_iterations + 1):
            start = i * BATCH_NUM
            end = min((i + 1) * BATCH_NUM, total_samples)
            
            if start == end:
                break
            
            my_bar.progress(end/total_samples, text=f"Processing samples {end} / {total_samples}")
            
            print("Processing " + str(start) + " to " + str(end) + "\n")
            cur_df = df.iloc[start:min(end, total_samples)]
            
            cur_res_df = pd.DataFrame(columns=['Input'] + labeler + ['Mode'])
            cur_res_df['Input'] = cur_df['Input']
            
            params = self.__getPredictParams(cur_df)
            
            success = []
            for model in labeler:
                try:
                    completion = await self.aiconfig.run(model, params=params)
                    pred_str = self.aiconfig.get_output_text(model)
                    pred_str = pred_str.replace('\'', '')
                    pred_str = pred_str.replace('\"', '')
                    predictions_lst = [x.strip() for x in pred_str.split(',')]
                    predictions_lst = [x.lower() for x in predictions_lst]
                    cur_res_df[model] = predictions_lst
                    success.append(model)
                except Exception as e:
                    print(e)
                    print("error in " + model + " output\n")
                            
            def custom_mode(df, model_list, default_label):
                model_results = [df[model] for model in model_list]
                counter = Counter(model_results)
                mode_of_results = mode(model_results)
                if counter[mode_of_results] == 1:
                    return default_label
                else:
                    return mode_of_results
            
            if calculate_mode:
                cur_res_df['Mode'] = cur_res_df.apply(lambda x: custom_mode(x, success, default_mode), axis=1)
                result_df = pd.concat([result_df, cur_res_df], ignore_index=True)

        return result_df

    def __getTrainParams(self, df, classes, task_name, input_desc, task_desc):
        """
        Return global parameter database.
        """
        assert (df.shape[0] <= 100)

        class_text = ', '.join(classes)
        classes_num = len(classes)

        params = {"task_name": task_name, "input_description": input_desc,
                  "task_description": task_desc,
                  "output_classes": class_text, "class_num": str(classes_num)}

        examples_text = ""
        for index, row in df.iterrows():
            examples_text = examples_text + \
                "Input: {{ input" + \
                str(index) + " }} ,Output: {{ output" + str(index) + " }}\n"
            params["input"+str(index)] = row[1]
            params["output"+str(index)] = row[0]

        params["examples"] = examples_text
        return params

    def __getPredictParams(self, df):
        """
        Return parameter to be used in predicting labels of testing dataset.
        """
        input_text = ""
        for i in range(df.shape[0]):
            input_text = input_text + str(i) + ". " + df.iloc[i, 0] + "\n"

        return {"real_inputs": input_text, "num_predictions": str(df.shape[0])}

