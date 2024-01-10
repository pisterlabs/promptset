import openai
import os
import tiktoken
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import math
import torch
from transformers import pipeline

import concurrent.futures
from transformers import pipeline
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]

tokenizer = tiktoken.get_encoding('cl100k_base')


# Utility functions
#  =================================================================
#  =================================================================
def count_tokens(text):
    token_count = len(tokenizer.encode(text))
    return token_count


def chunk_text(text, max_token_size):
    tokens = text.split(" ")
    token_count = 0
    chunks = []
    current_chunk = ""

    for token in tokens:
        token_count += count_tokens(token)

        if token_count <= max_token_size:
            current_chunk += token + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = token + " "
            token_count = count_tokens(token)

    if current_chunk:
        chunks.append(current_chunk.strip())
    print("chunks", len(chunks))
    return chunks

def tk_len(text):
    token = tokenizer.encode (
        text,
        disallowed_special=()
    )
    return len(token)

def split_chunks(text):
    chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=12)
    for chunk in splitter.split_documents(text):
        chunks.append(chunk)
    return chunks


def format_fixer(text):
    messages = [
        {"role": "system", "content": "You are a text formatting assistant."},
        {"role": "user", "content": f"Format the following text in a markdown file format. Maintain the rest of the details of the text as it is. Add line breaks after header and bullet points end: {text}"}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=messages,
        max_tokens=700,  # Adjust based on your desired summary length
        n=1,
        stop=None,
        temperature=0.1,
    )

    formatted_text = response.choices[0].message['content'].strip()
    print('summary-chunk', formatted_text)
    return formatted_text

def remove_conclusion(text):
    while True:
        # Find the start of the '## Conclusion' header
        header_start = text.find('## Conclusion')

        # If the header is not found, break the loop
        if header_start == -1:
            break

        # Find the end of the paragraph following the header
        paragraph_end = text.find('\n', header_start)

        # If the end of the paragraph is not found, set it to the end of the text
        if paragraph_end == -1:
            paragraph_end = len(text)

        # Remove the header and the following paragraph from the text
        text = text[:header_start] + text[paragraph_end+1:]

    return text

# Main functions
#  =================================================================
#  =================================================================
# def summarize_large_text(input_text, output_file):
#     # Chunk the text into smaller parts
#     input_text = input_text.replace('\n', '')
#     max_token_size = 3200 
#     text_chunks = chunk_text(input_text, max_token_size)
   
#     print("max token size", max_token_size)
#     # split_index = len(text_chunks) // 2
#     texts = [text_chunks[i:i+4] for i in range(0, len(text_chunks), 4)]
#     summaries = [generate_summary(chunk) for chunk in text_chunks]
#     # print(len(texts))
#     summaries_array = []
#     # # Generate summaries for each chunk concurrently
#     for text_array in texts:
#         with concurrent.futures.ThreadPoolExecutor() as executor:
#             summaries = list(executor.map(generate_summary, text_array))
#             summaries_array.append(summaries)
#             print('summaries',len(summaries))
   
    
#     # # Generate summaries for each chunk
#     # summaries = [generate_summary(chunk) for chunk in text_chunks]
#             print('summaries',len(summaries_array))
#             summaries_array = [item for sublist in summaries_array for item in sublist]


#     # Combine the summaries into a single article
#     article = "## Summary\n\n"
#     for idx, summary in enumerate(summaries_array, 1):
#         article +=  f" idx: {idx}:{summary}  \n\n"

#     token_limit = 3500
#     token_length = tk_len(article)
#     if int(token_length) > int(token_limit):
#         refinedSummary = refineSummaryChapters(article)
#     else:    
#         refinedSummary = refineSummary(article)
    
#     print('refined summary', refinedSummary)
#     # Save the article to a Markdown file
#     # with open(output_file, "w", encoding='utf-8') as f:
#     #     f.write(refinedSummary)
#     return refinedSummary



def summarize_large_text(input_text, summary_length, summary_format):
    # Chunk the text into smaller parts
    input_text = input_text.replace('\n', '')
    if(tk_len(input_text) < 14000):
        refinedSummary = refineSummary(input_text, summary_length, summary_format, long=True)
        response_obj = {
            'output_text': refinedSummary,
            'intermediate_steps': input_text 
        }     
    else:
        max_token_size = 3000
        text_chunks = chunk_text(input_text, max_token_size)
   
        print("max token size", max_token_size, tk_len(input_text))
    
        summaries = [generate_summary(chunk) for chunk in text_chunks]
        print('summaries',len(summaries))

        # # Combine the summaries into a single article
        article = "## Summary\n\n"
        for idx, summary in enumerate(summaries):  # enumerate starting from 1
            article +=  f" Part: {idx}:{summary}  \n\n"
        print('summaries_array', summary_length)    
        refinedSummary = refineSummary(article, summary_length, summary_format)
        
        response_obj = {
            'output_text': refinedSummary,
            'intermediate_steps': article
        }
         
    print('refined summary', response_obj)
 
    return response_obj

def generate_summary(text):
    print('summary-chunk length', int(tk_len(text)))
    text_token_length = tk_len(text)
    messages = [
        {"role": "system", "content": "You are a helpful assistant that extracts key information from text."},
        {"role": "user", "content": f"Take notes of the important points from the following text like a student would take in a lecture. Use subheaders and bullet points wherever required. Add line breaks after headers: \n\n\n{text}"}
    ]
  
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=500,  
        n=4,
        stop=None,
        temperature=0.1,
    )
    summary = response.choices[0].message['content'].strip()
    print('summary-chunk', summary)
    with open("workspace/chunks.txt", "a", encoding='utf-8') as f:
        f.write("\n\n\n" + summary)
    return summary


def refineSummary(text, summary_length, summary_format, long=False):
    output_format = 'article'
    
    if(summary_format == "paragraph"):
        output_format = 'paragraph'
    elif(summary_format == "ideas"):
        output_format = "key ideas"    
    else:
        output_format = "article"    

    print('used refine summary', summary_length, "format", output_format)

    # if prev_context:
    #     prompt = f"You are writing a summary of long text for me, here is the context of previous part of summary you wrote: {prev_context} \n\n. Now that you have the context, I will give you the next chunk of raw text. Generate summary of the raw text while maintaining the context in the form of a blog article in about 500 words. Add subheaders bullet points to make the article easily digestable. Maintain the context. Give the output in md format. Add line breaks after headers. Dont add a conclusion section since this is a middle part of long text summary. here is the text: \n\n {text}  "
    # else:
    prompt = f"this is the raw text that needs to be formatted in the form of a {output_format} in about {summary_length} words. Add a title to the output. Use subheaders and bullet points wherever required to make the article easily digestable. Maintain the context. Give the output in md format. Add line breaks after headers \n \n " + text
    print('refining summary', prompt)
    print('refining summary token count', tk_len(prompt))
   
    selected_model = "gpt-4"

    if (long == True):
        selected_model ="gpt-3.5-turbo-16k"
    
    completion = openai.ChatCompletion.create(
        model=selected_model,
        messages=[
            {"role": "system", "content": f"You are a helpful assistant that generates summary in form of {output_format} based on text provided"},
            {"role": "user", "content": prompt},         
        ],  
        n=1,
        stop=None,
        temperature=0.1,
    )    
    refined_summary = completion.choices[0].message.content
    print('refined summary', refined_summary)
    return refined_summary


def fixRawTextFormat(text, text_format):
    max_token_size = 8000
    text_chunks = chunk_text(text, max_token_size)

 
    formatted = [fixRawTextFormatChunk(chunk, text_format) for chunk in text_chunks]
    print('formats',len(formatted))

    # # Combine the formmated chunks into a single text file
    article = "\n\n"
    for formats in formatted:  # enumerate starting from 1
        article +=  f"{formats}  \n\n"

    return article
 

def fixRawTextFormatChunk(text, text_format):
    format_prompt = ''
    if text_format == 'text':
        format_prompt = "an article "
    else :    
        format_prompt = "an audio file transcript"
    prompt = f"this is section of a {format_prompt}. Your job is to fix the format of the text and output it as a readable content in markdown format using regular markdown symbols like # * etc \n \n " + text
    print('refining summary', prompt)
    print('refining summary token count', tk_len(prompt))
   
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates formatted text from the raw text provided"},
            {"role": "user", "content": prompt},         
        ],  
        n=1,
        stop=None,
        temperature=0.1,
    )    
    refined_summary = completion.choices[0].message.content
    return refined_summary

# Other unused functions
#  =================================================================
#  =================================================================
def summarize_large_text_langchain(input_text, max_token_size=13200):
    print('used langchain summary')   
    text_chunks = chunk_text(input_text, max_token_size)
    print('chunks', len(text_chunks))
    docs = [Document(page_content=t) for t in text_chunks]
    print('chunks', len(docs))
    prompt_template = """Take notes from the text in form of bullet points (maintain the context) output atleast 700-800 words. Extract as much information as possible that will be useful for summarising the text later: 


    {text}


    CONCISE SUMMARY:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    refine_template = (
        "this is the raw text that needs to be used to create a blog article which is a 10 min read. Add subheaders bullet points to make the article easily digestable. Maintain the context. Give the output in md format. Add line breaks after headers \n\n"
        " {existing_answer}\n"
        "------------\n"
        "{text}\n"
        "------------\n"
        
    )
    refine_prompt = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template,
    )
    llm = ChatOpenAI(temperature=0.1, max_tokens=2500, model_name="gpt-3.5-turbo-16k", openai_api_key=os.environ["OPENAI_API_KEY"])

    chain = load_summarize_chain(llm,
        chain_type="refine", return_intermediate_steps=True, question_prompt=PROMPT, refine_prompt=refine_prompt)
    results = chain({"input_documents": docs}, return_only_outputs=True)
    print(results)
    return results
    # with open(output_file, "w") as f:
    #     f.write(results['output_text'])


def bart_summariser(transcript):
    
    """
    It takes a transcript, tokenizes it, and then generates a summary using the BART model
    Args:
      transcript: The text you want to summarize.
    Returns:
      A summary of the text.
    """
    try:
        print("initiating summarizer...")
       
       
        tokenizer = AutoTokenizer.from_pretrained(
                "philschmid/bart-large-cnn-samsum")
        model = AutoModelForSeq2SeqLM.from_pretrained(
                "philschmid/bart-large-cnn-samsum")
        print("tokenizer and model were downloaded from huggingface")
        inputs = tokenizer(transcript,
                           max_length=1024,
                           truncation=True,
                           return_tensors="pt")
        summary_ids = model.generate(
            inputs["input_ids"], num_beams=8, min_length=200, max_length=5000)
        summary = tokenizer.batch_decode(
            summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        with open("workspace/chunks.txt", "a") as f:
            f.write(summary + "\n\n")
        print("summary generated", count_tokens(summary), summary)
        return summary
    except Exception as e:
        print("following error occured with bart summarised", e)        


def generate_context(text):
    prompt = f"Generate context of this text in not more than 400 words that would be helpful in summarising texts.  Dont add a conclusion section since this is a middle part of long text summary \n \n " + text
   
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates context of the text provided"},
            {"role": "user", "content": prompt},         
        ],  
        n=1,
        stop=None,
        temperature=0.1,
    )
    context = completion.choices[0].message.content
    print('generated context', context)
    return context

def refineSummaryChapters(text):
    max_token_size = 3000 
    text_chunks = chunk_text(text, max_token_size)

    refinedSummary = []
    prev_summary = refineSummary(text_chunks[0])
    refinedSummary.append(prev_summary)
    for chunk in text_chunks[1:]:
        # generate context from summary 
        context = generate_context(" ".join(refinedSummary))
        
        # refine text summary of chunks
        refined_intermediate = refineSummary(chunk, context)
        
        # append summary to refined summary array
        refinedSummary.append(refined_intermediate)

    final_refined_summary = "\n\n ".join(refinedSummary) 
    final_refined_summary = remove_conclusion(final_refined_summary)
    print("Final summary", final_refined_summary)
    return final_refined_summary    
     
            
       
