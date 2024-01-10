import os
import openai 
import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()  

def calculate_similarity(embeddings, input_embedding):
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    cosine_similarities = F.cosine_similarity(embeddings_tensor, input_embedding)
    return cosine_similarities

def get_sentence_embedding(sentence, tokenizer, model):
    encoded_input = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    return mean_pooling(model_output, encoded_input['attention_mask'])

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def get_openai_response(prompt, api_key):
    if api_key:
        openai.api_key = api_key
    else:
        openai.api_key = os.getenv('OPENAI_API_KEY')
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # gpt-3.5-turbo # gpt-4-1106-preview
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message['content'].strip()

def get_legislator_persona(user_problem, similar_laws, api_key):
    prompt = f"""사용자가 제출한 문제와 식별된 상위 5개의 유사한 법률을 기반으로 이 문제를 챔피언할 국회의원의 페르소나를 만듭니다. 
    그들의 정책 방향, 공약 및 유권자에 대한 어필 전략을 포함하세요. 
    페르소나가 시민의 필요와 제공된 입법적 맥락과 조화를 이루도록 합니다.\n\n사용자 문제: {user_problem}\n식별된 유사한 법률: {similar_laws}\n\n입법 환경에서 이러한 문제를 효과적으로 옹호할 수 있는 필요한 특성을 가진 페르소나를 생성하세요. 이름은 생성하면 안됨.
    아래 예시를 참고해서 작성하세요.
    ### 예시
    국회의원 페르소나:
    관련 부처:
    제안 법안 명칭:
    페르소나의 목적:
    """
    persona = get_openai_response(prompt, api_key)
    return persona

def main():
    st.title("AI-based Legislation Drafting and Similarity Checker")
    api_key = st.text_input("Enter your OpenAI GPT API key:", type="password")

    user_problem = st.text_area("Describe your problem or situation for the legislation:")
    
    if 'generated_context' not in st.session_state:
        if st.button('Generate Legislation Context', key='generate_context'):
            if user_problem:
                prompt = f"""너는 한국어로 {user_problem}값을 받아서 국회 법안을 발의 하는 ai야. 너가 뭘했다라고 먼저 얘기하지 말고 보고서 형식으로만 출력해. A4 한장 분량으로 {user_problem}을 국회에 입법하게 할 법안으로 만들어줘. 들어가야하는 내용은 법안의 제목, 목적, 내용, 발의의 이유 및 원인, 해결방안, 앞으로의 전략 등이야.
                    아래 예시를 참고해서 작성하세요.
                    ### 예시
                    보고서 제목:
                    보고서 작성일자:
                    관련 부처:
                    제안 법안 명칭: 
                    법안 목적:
                    법안 내용:
                    1.
                    2.
                    3.
                    ..
                    발의의 이유 및 원인: 
                    해결 방안: 
                    앞으로의 전략: """
                context = get_openai_response(prompt, api_key)
                st.session_state['generated_context'] = context
                st.markdown("#### Generated context for legislation:")
                st.markdown(context, unsafe_allow_html=True)
            else:
                st.warning("Please enter a problem description first.")
    else:
        st.markdown("#### Generated context for legislation:")
        st.markdown(st.session_state['generated_context'], unsafe_allow_html=True)

    if 'generated_context' in st.session_state and st.button('Find Similar Laws'):
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

        df = pd.read_csv('web/embeddings_data.csv')
        db_embeddings = torch.stack(tuple(df['Embedding'].apply(lambda emb: torch.tensor(eval(emb), dtype=torch.float))))

        context_embedding = get_sentence_embedding(st.session_state['generated_context'], tokenizer, model).unsqueeze(0)

        cosine_sim = calculate_similarity(db_embeddings, context_embedding)

        top_indices = cosine_sim.topk(5).indices.numpy()

        similar_laws_info = []
        for index in top_indices:
            law_info = df.iloc[index][['제목', '내용']].to_dict()
            similar_laws_info.append(f"{law_info['제목']} - {law_info['내용']}")

        st.session_state['similar_laws_info'] = similar_laws_info
        st.session_state['similar_laws_display'] = []

        for index in top_indices:
            law_info = df.iloc[index][['제목', '내용']]
            law_info['제목'] = law_info['제목'].replace('\\n', '\n')
            law_info['내용'] = law_info['내용'].replace('\\n', '\n')
            st.session_state['similar_laws_display'].append(law_info)

    if 'similar_laws_display' in st.session_state:
        for law_info in st.session_state['similar_laws_display']:
            st.write(law_info)
            st.write("------")

    if 'similar_laws_info' in st.session_state:
        if st.button('Create Legislator Persona', key='create_persona'):
            legislator_persona = get_legislator_persona(user_problem, '\n'.join(st.session_state['similar_laws_info']), api_key)
            st.subheader("국회의원 페르소나")
            st.markdown(legislator_persona, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
