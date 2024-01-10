from openai.embeddings_utils import distances_from_embeddings
import numpy as np
import pandas as pd
import openai

import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("API_KEY")

print(openai.api_key)

INPUT_FILE = 'processed/vn_embeddings_200_200_100.csv'

# df=pd.read_csv('processed/en_embeddings.csv', index_col=0)


def create_context(
    question: str,
    df: pd.DataFrame,
    max_len,
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']


    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')

    returns = []
    cur_len = 0
    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():
        
        # Add the length of the text to the current length
        cur_len += row['n_tokens']
        
        # If the context is too long, break
        if cur_len > max_len:
            break
        
        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)

def answer_question(
    df: pd.DataFrame,
    question: str,
    max_len=1500,
    debug=False,
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question = question,
        max_len=max_len,
        df=df,
    )
    
    prompt_en = f"""
        Answer the question based on the context below. Focus on the informations about numbers, objects, actions in the questions. 
        If the question can't be answered based on the context, say \"I don't know\"

        ---

        Context: {context}

        ---

        Question: {question}
    """

    prompt_vn = f"""
        Trả lời câu hỏi dựa vào đoạn dữ liệu dưới đây. Nếu không tìm thấy câu trả lời, trả về \"Tôi không tìm thấy thông tin này\"

        ---

        Dữ liệu: {context}

        ---

        Câu hỏi: {question}
    """

    if (debug):
        print(prompt_vn)
    try:
        # Create a completions using the question and context
        response = openai.ChatCompletion.create(
            temperature=0,
            max_tokens=300,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are helpful assistant."},
                {"role": "user", "content": prompt_vn}
            ]
        )

        return response.choices[0]["message"]["content"]
    except Exception as e:
        print(e)
        return ""

if __name__ == '__main__':
    # df=pd.read_csv('processed/vn_embeddings.csv', index_col=0)
    INPUT_FILE = 'processed/vn_embeddings_200_200_100.csv'
    df=pd.read_csv(INPUT_FILE, index_col=0)

    df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)
    print(answer_question(question= 'thành tựu của Cohost AI qua các năm?', debug=True, df = df))
    # print(answer_question(question='List all positions the company is looking for.', debug=True, df = df))