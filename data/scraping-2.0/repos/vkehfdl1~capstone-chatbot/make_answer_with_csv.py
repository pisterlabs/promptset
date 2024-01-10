import click
import pandas as pd
from dotenv import load_dotenv
from langchain.llms.vllm import VLLMOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser

load_dotenv()

prompt = PromptTemplate.from_template(
    """
    Answer user’s question about NLP paper using given paper passages.
    
    Question: {question}
    
    Paper passages:
    {passage_1}
    {passage_2}
    {passage_3}
    {passage_4}
    {passage_5}
    
    Answer:
    """
)
runnable = prompt | VLLMOpenAI(model_name="meta-llama/Llama-2-7b-hf",
                               openai_api_base="https://e65a-34-143-245-217.ngrok-free.app/v1",
                               openai_api_key="") | StrOutputParser()


def make_answer_pd(row):
    answer = runnable.invoke({
        "question": row['question'],
        # "passage_1": row['passage_content_1'],
        # "passage_2": row['passage_content_2'],
        # "passage_3": row['passage_content_3'],
        # "passage_4": row['passage_content_4'],
        # "passage_5": row['passage_content_5'],
        "passage_1": "",
        "passage_2": "",
        "passage_3": "",
        "passage_4": "",
        "passage_5": ""
    })
    return answer


@click.command()
@click.option('--path', help='csv file path')
@click.option('--save_path', help='save path')
def main(path, save_path):
    df = pd.read_csv(path)
    df['answer'] = df.apply(make_answer_pd, axis=1)
    df.to_csv(save_path, index=False)


if __name__ == "__main__":
    main()
