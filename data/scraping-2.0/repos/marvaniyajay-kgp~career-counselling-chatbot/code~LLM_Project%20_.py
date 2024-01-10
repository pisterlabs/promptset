#!/usr/bin/env python
# coding: utf-8

# In[13]:


from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
import sys
import os


# In[ ]:


os.environ["OPENAI_API_KEY"] = "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"


# In[ ]:


def createVectorIndex(path):
    max_input = 4096
    tokens = 256
    chunk_size = 600
    max_chunk_overlap = 20
    
    prompt_helper = PromptHelper(max_input, tokens, max_chunk_overlap, chunk_size-limit = chunk_size)
    
    #Define LLM
    llmPredictor = LLMPredictor(llm = OpenAI(temperature = 0, model_name = "text-ada-001", max_tokens =tokens))
    
    #load data
    docs = SimpleDirectoryReader(path).load_data()
    
    #Create Vector Index
    vectorIndex = GPTSimpleVectorIndex(documents = docs, llm_predictor = llmPredictor, prompt_helper = prompt_helper)
    vectorIndex.save_to_disk('vectorIndex.json')
    return vectorindex

vectorIndex = createVectorIndex('Knowledge')


# In[ ]:


def answerMe(vectorIndex):
    vIndex = GPTSimpleVectorIndex.load_from_disc(vectorIndex)
    while True:
        prompt = input('Please ask:')
        response = vIndex.query(prompt,response_mode = "compact")
        print(f"Response: {response} \n")
        


# In[ ]:


answerMe('vectorIndex.json')


# Certainly! Here is the gist of the impact and process taken to create the chatbot for JOSAA counseling:
# 
# 1. Libraries:
#    - `gpt_index`: Used for creating a vector index and querying the index.
#    - `langchain`: Provides access to the OpenAI language model for text generation.
#    - `sys`: Provides access to system-specific parameters and functions.
#    - `os`: Provides a way to use operating system-dependent functionality.
# 
# 2. Environment setup:
#    - The OpenAI API key is stored in the `OPENAI_API_KEY` environment variable.
# 
# 3. `createVectorIndex` function:
#    - This function creates a vector index using the `GPTSimpleVectorIndex` class.
#    - It takes a `path` parameter representing the directory where the knowledge documents are stored.
#    - The function uses the `SimpleDirectoryReader` to load the data from the specified directory.
#    - It sets up a language model (LLM) using the `OpenAI` class and creates an instance of `LLMPredictor`.
#    - The `PromptHelper` class is used to handle prompts and chunking of input text.
#    - The `GPTSimpleVectorIndex` is instantiated with the loaded documents, LLM predictor, and prompt helper.
#    - The vector index is saved to disk as 'vectorIndex.json' and returned.
# 
# 4. `answerMe` function:
#    - This function takes the path to the vector index file as a parameter.
#    - The vector index is loaded from the specified file using the `GPTSimpleVectorIndex.load_from_disk` method.
#    - It enters a loop where it prompts the user for input using `input('Please ask:')`.
#    - The vector index queries the input prompt using `vIndex.query` with the response mode set to "compact".
#    - The response from the vector index is printed to the console.
# 
# 5. Execution:
#    - The `createVectorIndex` function is called with the 'Knowledge' directory path to create the vector index.
#    - The `answerMe` function is called with the path to the vector index file to start the interactive question-answering loop.
# 
# To make this code CV-fit for a Data profile and pass ATS (Applicant Tracking System), you can follow these suggestions:
# - Add a header at the beginning of the code with your name, contact information, and a brief summary of the purpose of the code.
# - Include relevant import statements for libraries commonly used in a Data profile, such as `numpy`, `pandas`, and `scikit-learn`.
# - Use meaningful variable and function names that reflect the purpose of the code and follow standard naming conventions.
# - Add comments throughout the code to explain the different sections and provide clarity on the steps taken.
# - Use docstrings to provide descriptions of functions and their parameters.
# - Consider modularizing the code into separate functions or classes to improve code organization and maintainability.
# - Ensure the code follows PEP 8 style guidelines for Python code.
# - Include relevant information about any machine learning models or algorithms used, along with their purpose and impact.
# 
# By incorporating these suggestions, you can make the code more presentable and tailored for a Data profile.
# 
# In the above project, the LLM Predictor plays a crucial role in generating responses for the chatbot. The LLM Predictor, represented by the `LLMPredictor` class, utilizes OpenAI's Language Model (LLM) through the `langchain` library.
# 
# The main function of the LLM Predictor is to interface with the LLM model and generate contextually relevant responses based on user queries. It takes user input or prompts as input and passes them to the LLM model for processing. The LLM Predictor configures the LLM with specific settings such as temperature, model name, and maximum tokens to control the response generation process.
# 
# The LLM Predictor leverages the power of the LLM model, which is trained on a vast amount of textual data, to generate intelligent and coherent responses. It uses the learned patterns, semantics, and language understanding capabilities of the LLM to generate contextually appropriate and informative responses to user queries.
# 
# By utilizing the LLM Predictor, the chatbot can provide accurate and relevant information to users during JOSAA counselling, enhancing the user experience and facilitating effective communication between the chatbot and users.
# 
# OpenAI's Language Model (LM) is based on deep learning techniques, particularly transformer neural networks. The model is trained on a large corpus of text data, such as books, articles, and web pages, to learn the statistical patterns and structure of human language.
# 
# The language model works by predicting the likelihood of the next word or sequence of words given the previous context. It learns to capture the relationships between words, their meanings, and the syntactic and semantic structures of sentences.
# 
# During training, the model processes the input text in a sequential manner, taking into account the context of each word or token. It assigns probabilities to different possible next words based on the preceding context and adjusts its internal parameters through a process called backpropagation, optimizing the model to minimize prediction errors.
# 
# When generating text, the language model utilizes this learned knowledge to predict the most likely sequence of words given an input prompt or context. It can generate coherent and contextually relevant responses by leveraging the statistical patterns and associations it has learned from the training data.
# 
# OpenAI's language model, such as GPT-3 or GPT-4, is designed with a vast number of parameters, enabling it to capture complex linguistic patterns and produce high-quality text generation. The model's ability to understand and generate human-like text has various applications, including chatbots, language translation, content generation, and more.
# 
# Certainly! The process of transforming user input into a response output involves several steps:
# 
# 1. User Input: The process starts when the user provides input to the system, typically in the form of a prompt or a question.
# 
# 2. Preprocessing: The input is preprocessed to ensure it is in a suitable format for the language model. This may involve tasks such as tokenization, which breaks the input text into individual units (tokens) that the model can understand.
# 
# 3. Encoding: The preprocessed input is encoded into a numerical representation that the language model can process. This encoding captures the semantic meaning and contextual information of the input.
# 
# 4. Passing through the Model: The encoded input is fed into the language model, such as OpenAI's LLM, which processes the input using its deep learning architecture. The model analyzes the input in the context of its learned knowledge and statistical patterns.
# 
# 5. Prediction: The language model predicts the most likely next words or sequence of words based on the encoded input. It generates a response that is coherent and contextually relevant to the input.
# 
# 6. Decoding: The predicted output from the language model is decoded from its numerical representation back into human-readable text.
# 
# 7. Postprocessing: The decoded response may undergo postprocessing to refine and enhance its quality. This can involve tasks such as removing unnecessary information or adding formatting.
# 
# 8. Output: The final response is presented to the user as the output of the system, providing an answer or information based on the user's input.
# 
# Throughout this process, the language model leverages its training on a large corpus of text data to generate meaningful and contextually appropriate responses. The transformation from user input to predictor response involves encoding, prediction, and decoding steps that utilize the model's understanding of language and its statistical patterns.
# 
# Certainly! The process of encoding involves converting the user input into a numerical representation that the language model can understand and process. Here's a detailed explanation of the encoding process:
# 
# 1. Tokenization: The first step in encoding is tokenization. Tokenization breaks down the user input, whether it's a sentence, phrase, or paragraph, into individual units called tokens. Tokens can be as small as individual words or even smaller units like subwords or characters. The purpose of tokenization is to create a standardized representation of the text that can be processed by the language model.
# 
# 2. Vocabulary Mapping: Once the input is tokenized, each token needs to be mapped to a unique identifier or index. This mapping is typically done using a vocabulary, which is a collection of all the unique tokens present in the language model. Each token is assigned a specific index value from the vocabulary.
# 
# 3. Embedding: After mapping the tokens to their respective indices, the next step is to convert these indices into dense numerical vectors called embeddings. Embeddings capture the semantic meaning and contextual information of the tokens. These dense representations allow the language model to understand relationships between tokens and their meanings. The embedding process can involve techniques like word embeddings (e.g., Word2Vec or GloVe) or subword embeddings (e.g., BytePair Encoding or WordPiece).
# 
# 4. Sequence Encoding: If the input consists of multiple tokens arranged in a sequence, such as a sentence or paragraph, sequence encoding is applied. This step captures the order and sequential structure of the tokens. Techniques like recurrent neural networks (RNNs), long short-term memory (LSTM), or transformers are commonly used for sequence encoding.
# 
# 5. Padding and Truncation: To ensure consistent input size, padding or truncation may be applied. Padding adds special tokens (e.g., zeros) to the input sequence to match a fixed length, while truncation removes excess tokens if the input sequence is longer than the desired length.
# 
# The encoded representation of the user input, which includes token indices, embeddings, and sequence encoding, is then passed into the language model for processing. The model utilizes this numerical representation to generate responses based on the learned patterns and contextual information.
# 
# By encoding the user input, the language model can understand and process the input text effectively, enabling it to generate meaningful and contextually relevant responses.

# In[ ]:




