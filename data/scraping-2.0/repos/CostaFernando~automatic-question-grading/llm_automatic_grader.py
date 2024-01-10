import boto3
import json
import pandas as pd
import re
from tqdm import tqdm
from openai import OpenAI
import openai
import backoff

open_ai_client = OpenAI(api_key="OPENAI_API_KEY")

bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')

def extract_first_one_digit_number(text):
    match = re.search(r'\b\d\b', text)
    return match.group(0) if match else None

def llama2_chat_inference(instruction_prompt_initial, instruction_prompt_final, question, correct_answer, student_answer, model_id, max_gen_len=50, temperature=0.0, top_p=1.0):
    prompt = f"""
    [INST]{instruction_prompt_initial}[/INST]
    [INST]Questão:
    {question}
    Gabarito:
    {correct_answer}
    [/INST]
    [INST]Resposta do aluno:
    {student_answer}
    [/INST]
    [INST]{instruction_prompt_final}[/INST]
    """

    body = json.dumps({
        "prompt": prompt,
        "max_gen_len": max_gen_len,
        "temperature": temperature,
        "top_p": top_p,
    })

    response = bedrock.invoke_model(body=body, modelId=model_id, accept='application/json', contentType='application/json')

    response_body = json.loads(response.get('body').read())

    return response_body["generation"]

@backoff.on_exception(backoff.expo, openai.RateLimitError)
def open_ai_gpt_inference(instruction_prompt_initial, instruction_prompt_final, question, correct_answer, student_answer, model_id, max_gen_len=30, temperature=0.0, top_p=1.0):
    response = open_ai_client.chat.completions.create(
      model=model_id,
      messages=[
        {
          "role": "system",
          "content": instruction_prompt_initial
        },
        {
          "role": "user",
          "content": f"Questão:\n{question}\n\nGabarito:\n{correct_answer}"
        },
        {
          "role": "user",
          "content": f"Resposta do aluno:\n{student_answer}"
        },
        {
          "role": "system",
          "content": instruction_prompt_final
        }
      ],
      temperature=temperature,
      max_tokens=max_gen_len,
      top_p=top_p,
      frequency_penalty=0,
      presence_penalty=0
    )
    
    return response.choices[0].message.content

student_answers_df = pd.read_csv('Syn data alg progr - answers.csv')
questions_df = pd.read_csv('Syn data alg progr - questions.csv')

pivot_df = pd.merge(questions_df, student_answers_df, on='Question Number')
pivot_df = pivot_df[['Question Number', 'Question', 'Correct Answer', 'Synthetic Answer']]

instruction_prompt_initial = """
Você é um assistente de professor. Você corrigirá questões respondidas por alunos, contrastando as respostas deles com o gabarito da questão.

Considere a seguinte escala de possíveis notas para avaliar a resposta do aluno:
1 = Insuficiente (totalmente errado)
2 = Básico (parcialmente errado)
3 = Proficiente (totalmente certo)
"""
instruction_prompt_final = """
Retorne apenas a nota com o número 1, 2 ou 3.
"""

model_inference_function_map = {
    'meta.llama2-13b-chat-v1': llama2_chat_inference,
    'meta.llama2-70b-chat-v1': llama2_chat_inference,
    'gpt-3.5-turbo-1106': open_ai_gpt_inference,
    'gpt-4-1106-preview': open_ai_gpt_inference
}

model_id = 'meta.llama2-13b-chat-v1'
model_inference_function = model_inference_function_map[model_id]

model_grades = []

for index, row in tqdm(pivot_df.iterrows(), total=pivot_df.shape[0], desc="Processing questions"):
    question_number = row['Question Number']
    question = row['Question']
    correct_answer = row['Correct Answer']
    student_answer = row['Synthetic Answer']

    response = model_inference_function(instruction_prompt_initial, instruction_prompt_final, question, correct_answer, student_answer, model_id)
    print(response)
    predicted_grade = extract_first_one_digit_number(response)

    model_grades.append({'Question Number': question_number, 'Model Grade': predicted_grade})

results_df = pd.DataFrame(model_grades)
results_df.to_csv(f'{model_id}_grades.csv', index=False)