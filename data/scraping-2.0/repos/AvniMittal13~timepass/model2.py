from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers, HuggingFacePipeline
from langchain.chains import RetrievalQA
import chainlit as cl
import torch
from torch import cuda, bfloat16
import transformers
from transformers import StoppingCriteria, StoppingCriteriaList


DB_FAISS_PATH = 'vectorstore/db_faiss'

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, return an empty list, don't try to make up an answer.

Given a user query in English, you will respond with a well-reasoned response in JSON format using a set of unknown tools.
```json
[
    {
        "tool_name": "TOOL_NAME",
        "arguments": [
            {
                "argument_name": "ARGUMENT_NAME",
                "argument_value": "ARGUMENT_VALUE"
            }
        ]
    }
]
```
Please provide the necessary tool arguments in the JSON format specified.

Context: {context}
Question: {question}
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain


# define custom stopping criteria object



def create_pipeline():
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

    model = transformers.AutoModelForCausalLM.from_pretrained(
        'mosaicml/mpt-7b-instruct',
        trust_remote_code=True,
        torch_dtype=bfloat16,
        max_seq_len=2048
    )
    model.eval()
    model.to(device)
    print(f"Model loaded on {device}")

    tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    stop_token_ids = tokenizer.convert_tokens_to_ids(["<|endoftext|>"])

    class StopOnTokens(StoppingCriteria):
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            for stop_id in stop_token_ids:
                if input_ids[0][-1] == stop_id:
                    return True
            return False
        
    stopping_criteria = StoppingCriteriaList([StopOnTokens()])

    generate_text = transformers.pipeline(
        model=model, tokenizer=tokenizer,
        return_full_text=True,  # langchain expects the full text
        task='text-generation',
        device=device,
        # we pass model parameters here too
        stopping_criteria=stopping_criteria,  # without this model will ramble
        temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
        top_p=0.15,  # select from top tokens whose probability add up to 15%
        top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
        max_new_tokens=64,  # mex number of tokens to generate in the output
        repetition_penalty=1.1  # without this output begins repeating
    )

    return generate_text


#Loading the model
def load_llm():
    # Load the locally downloaded model here
    generate_text = create_pipeline()
    # llm = CTransformers(
    #     model = "TheBloke/Llama-2-7B-Chat-GGML",
    #     model_file ="llama-2-7b-chat.ggmlv3.q8_0.bin",
    #     max_new_tokens = 512,
    #     temperature = 0.5
    # )
    llm = HuggingFacePipeline(pipeline=generate_text)

    return llm

#QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

#output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

#chainlit code
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to DevRev Bot. What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer += "\nNo sources found"

    await cl.Message(content=answer).send()


"""
<TOOL>
{
    "tool_name": "prioritize_objects",
    "description" : "The tool generates a sorted list of objects based on an internal priority logic, the details of which are unspecified. It takes a single argument, 'objects', which should be provided as an array of objects. The tool is designed to offer a straightforward means of organizing and ranking objects without disclosing the specific criteria used for prioritization.",
    "tool_description": "Returns a list of objects sorted by priority. The logic of what constitutes priority for a given object is an internal implementation detail.",
    "arguments": [
        {
            "arguments_name": "objects",
            "arguments_description": "A list of objects to be prioritized",
            "arguments_type": "array of objects"
        }
    ]
}
</TOOL>
<TOOL>
{
    "tool_name": "prioritize_objects",
    "prioritize_objects" : "The tool generates a sorted list of objects based on an internal priority logic, the details of which are unspecified. It takes a single argument, 'objects', which should be provided as an array of objects. The tool is designed to offer a straightforward means of organizing and ranking objects without disclosing the specific criteria used for prioritization.",
    "tool_description": "Returns a list of objects sorted by priority. The logic of what constitutes priority for a given object is an internal implementation detail.",
    "arguments": [
        {
            "arguments_name": "objects",
            "arguments_description": "A list of objects to be prioritized",
            "arguments_type": "array of objects"
        }
    ]
}
</TOOL>
"""