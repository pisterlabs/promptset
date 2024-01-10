# -*- coding: utf-8 -*-
# This script is used to evaluate the performance of Pigeon AI Video Translation system by using Large Language Model.

# Written by Jiaen LIU, 2023/09/18

# Import the necessary packages
import re

from langchain.evaluation import load_evaluator, EvaluatorType
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
# from src.srt_util.srt import SrtScript

# Load the evaluator

def init_evaluator(source_lang="en", target_lang="zh", domain="startcraft2", model="gpt-4-0613"):

    # map the language code to the language name
    language_map = {
        "en": "English",
        "zh": "Chinese",
    }

    llm = ChatOpenAI(temperature=0, model=model)

    # Completeness is the percentage of the input that is translated, to test if there is any missing information
    # Accuracy is the percentage of the translation that is correct
    fstring = """
            You are grading the translation based on following input:
            {input}
            if the input is "", that means there is no input sentence.
            you should grade the translation based on the reference translation:
            Here is the real answer(reference):
            {reference}
            You are grading the following translation:
            {output}
            based on the following criteria: 
            {criteria}
            Give two grades, accuracy and completeness rate them from a scale of 0 to 100, where 0 is the lowest (very low accuracy/completeness) and 100 is the highest (very high accuracy/completeness)? 
            Give explanations for every single one and if the answer if partially correct that is acceptable. However punish the scores for answers that are 
            numerically incorrect this also includes values that have the $ in front
            Please give the completeness score first followed by the accuracy score. 
            For example: 
            Accuracy: 40. Explanation here
            Completeness: 80. Explanation here
            Do not differ from the format ever
            """
    
    if source_lang in language_map and target_lang in language_map:
        lang_str = f"You are an expert {language_map[source_lang]} to {language_map[target_lang]} translator specialized in {domain}."
        prompt = PromptTemplate.from_template(lang_str+fstring, template_format="f-string")
    
    else:
        print("The language code is not supported, please check the language code.")
        prompt = PromptTemplate.from_template(fstring, template_format="f-string")

    return load_evaluator("labeled_criteria", llm=llm, prompt=prompt, criteria="correctness")

# prase the output of the evaluation
# example : 
# 'value': 'Accuracy: 80. The predicted answer is partially correct. The sentence "这是一个测试句子" translates to "This is a test sentence" in English. However, the original sentence is "This is an test sentences" which is grammatically incorrect in English. The correct translation should be "这是一个测试句子" if we correct the English sentence to "This is a test sentence". Therefore, the predicted answer is not entirely wrong, but it does not match the original sentence exactly due to the grammatical error in the original sentence.'
# def parse_eval_result(eval_result):
#     # score = eval_result.score
#     value = eval_result["value"]
#     value = value.split("Accuracy: ")[1].split(".")
#     # combine the rest of the string into the whole explanation
#     explanation = ".".join(value[1:])
#     return int(value[0]), explanation

# def parse_eval_result(eval_result):
#     # Extract the 'Accuracy' score using a regular expression from the 'reasoning' key
#     accuracy_match = re.search(r'Accuracy: (\d+)', eval_result['value'])
#     print(accuracy_match)
#     if accuracy_match:
#         accuracy = int(accuracy_match.group(1))
#     else:
#         # try to get the accuracy from the 'value' key
#         accuracy = 0

#     # Directly get the 'Explanation' value from the 'value' key
#     explanation = eval_result['value']

#     return accuracy, explanation

def parse_eval_result(data):
    # Extract the value string
    value_str = data.get('value', '').lower()
    reasoning_str = data.get('reasoning', '').lower()

    response = value_str + reasoning_str

    # Use regex to extract accuracy value and explanation
    accuracy_match = re.search(r'accuracy: (\d+)', response)
    acc_explanation_match = re.search(r'accuracy: \d+\. (.+)', response)

    # Use regex to extract completeness value and explanation
    completeness_match = re.search(r'completeness: (\d+)', response)
    completeness_explanation_match = re.search(r'completeness: \d+\. (.+)', response)

    # Extract the matched groups
    completeness = int(completeness_match.group(1)) if completeness_match else None
    completeness_explanation = completeness_explanation_match.group(1) if completeness_explanation_match else None
    accuracy = int(accuracy_match.group(1)) if accuracy_match else None
    acc_explanation = acc_explanation_match.group(1) if acc_explanation_match else None

    return (accuracy, acc_explanation), (completeness, completeness_explanation)

def evaluate_prediction(input, reference, prediction, evaluator):
    eval_result = evaluator.evaluate_strings(
        prediction=prediction,
        input=input,
        reference=reference,
    )
    print(eval_result)
    return parse_eval_result(eval_result)

if __name__ == "__main__":
    evaluator = init_evaluator()
    # For no input english sentence, just put "" in the input
    accuracy, completeness = evaluate_prediction("it's obviously going to be 神族 trying to go for a 野炮台", " 每当我们看到BF开", " 每当我们看到BF开", evaluator)
    print("Accuracy:", accuracy[0])
    print("Acc_Explanation:", accuracy[1])
    print("Completeness:", completeness[0])
    print("Comp_Explanation:", completeness[1])
