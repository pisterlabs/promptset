import json

chapter = {"chapters" : [ {
  "chapterId": 1,
  "chapterTitle": "What are Large Language Models?",
  "sections": [
    {
      "sectionId": 1,
      "sectionTitle": "Introduction to Large Language Models",
      "content": "Large language models (LLMs) are advanced artificial intelligence algorithms trained on massive amounts of data. They're based on transformer architectures and can understand and generate human language."
    },
    {
      "sectionId": 2,
      "sectionTitle": "Working of Large Language Models",
      "content": "LLMs are usually provided as a service over an API or web interface. LLMs can understand multiple languages and various topics, enabling them to produce text in different styles.",
      "subsections": [
        {
          "subsectionId": 1,
          "subsectionTitle": "Architecture",
          "content": "LLMs are based on the transformer model architecture, which includes multiple layers of self-attention mechanisms."
        },
        {
          "subsectionId": 2,
          "subsectionTitle": "Training",
          "content": "The GPT-3 model, for example, was trained on vast amounts of text data from the internet, which helps it generate coherent and contextually-relevant responses."
        }
      ]
    },
    {
      "sectionId": 3,
      "sectionTitle": "Applications of Large Language Models",
      "content": "Large language models can be used for tasks such as text generation, summarization, translation, and sentiment analysis. They have revolutionized the field of conversational AI and have real-world applications in industries and businesses, such as support chatbots for customer engagement."
    }
  ]
},

{
  "chapterId": 2,
  "chapterTitle": "Intro to Langchain",
  "sections": [
    {
      "sectionId": 1,
      "sectionTitle": "What is Langchain?",
      "content": "Langchain is a powerful tool for working with large language models (LLMs) that simplifies the process of composing these pieces and provides an abstraction for building custom knowledge chatbots. It works by taking a large source of data, breaking it down into chunks, and embedding them into a Vector Store. When a prompt is inserted into the chatbot, Langchain queries the Vector Store for relevant information, which is then used in conjunction with the LLM to generate the answer."
    },
    {
      "sectionId": 2,
      "sectionTitle": "Why Do We Need Langchain?",
      "content": "Langchain offers a useful approach to overcome the limitations of LLMs by preprocessing the corpus of text, breaking it down into chunks or summaries, embedding them in a vector space, and searching for similar chunks when a question is asked. This pattern of preprocessing, real-time collecting, and interaction with the LLM is common and can be used in other scenarios, such as code and semantic search. Langchain provides an abstraction that simplifies the process of composing these pieces, making it easier to work with large language models."
    },
    {
      "sectionId": 3,
      "sectionTitle": "Example: Building a Question-Answering App with Langchain",
      "content": "Let's build a simple question-answering app using Langchain.",
      "subsections": [
        {
          "subsectionId": 1,
          "subsectionTitle": "Step 1: Install Langchain",
          "code": "pip install langchain"
        },
        {
          "subsectionId": 2,
          "subsectionTitle": "Step 2: Import required libraries",
          "code": "import langchain as lc\nfrom langchain import SimpleSequentialChain"
        },
        {
          "subsectionId": 3,
          "subsectionTitle": "Step 3: Load a large language model",
          "code": "model = lc.load('gpt-3')"
        },
        {
          "subsectionId": 4,
          "subsectionTitle": "Step 4: Define a function to answer questions",
          "code": "def get_answer(prompt):\n  chain = SimpleSequentialChain(model)\n  chain.add_prompt(prompt)\n  response = chain.generate()\n  return response"
        },
        {
          "subsectionId": 5,
          "subsectionTitle": "Step 5: Get answers to your questions",
          "code": "question = 'What is the capital of France?'\nanswer = get_answer(question)\nprint(answer)"
        }
      ]
    }
  ]
}
]}




# Writing to sample.json 
with open('chapters.json', 'w') as json_file:
    json.dump(chapter, json_file)
