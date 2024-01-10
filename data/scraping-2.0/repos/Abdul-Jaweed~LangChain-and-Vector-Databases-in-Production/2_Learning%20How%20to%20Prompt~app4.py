## Chain of Thought Prompting

# Chain of Thought Prompting (CoT) is a technique developed to encourage large language models to explain their reasoning process, leading to more accurate results. By providing few-shot exemplars demonstrating the reasoning process, the LLM is guided to explain its reasoning when answering the prompt. This approach has been found effective in improving results on tasks like arithmetic, common sense, and symbolic reasoning.

# In the context of LangChain, CoT can be beneficial for several reasons. First, it can help break down complex tasks by assisting the LLM in decomposing a complex task into simpler steps, making it easier to understand and solve the problem. This is particularly useful for calculations, logic, or multi-step reasoning tasks. Second, CoT can guide the model through related prompts, helping generate more coherent and contextually relevant outputs. This can lead to more accurate and useful responses in tasks that require a deep understanding of the problem or domain.

# There are some limitations to consider when using CoT. One limitation is that it has been found to yield performance gains only when used with models of approximately 100 billion parameters or larger; smaller models tend to produce illogical chains of thought, which can lead to worse accuracy than standard prompting. Another limitation is that CoT may not be equally effective for all tasks. It has been shown to be most effective for tasks involving arithmetic, common sense, and symbolic reasoning. For other types of tasks, the benefits of using CoT might be less pronounced or even counterproductive.

# Tips for Effective Prompt Engineering
    # Be specific with your prompt: Provide enough context and detail to guide the LLM toward the desired output.
    # Force conciseness when needed.
    # Encourage the model to explain its reasoning: This can lead to more accurate results, especially for complex tasks.

# Keep in mind that prompt engineering is an iterative process, and it may require several refinements to obtain the best possible answer. As LLMs become more integrated into products and services, the ability to create effective prompts will be an important skill to have.





from langchain import FewShotPromptTemplate, PromptTemplate, LLMChain
from langchain.llms import OpenAI

import os
from dotenv import load_dotenv

load_dotenv()

apikey = os.getenv("OPENAI_API_KEY")

# Initialize LLM
llm = OpenAI(
    llm=llm,
    model="text-davinci-003",
    temperature=0
)

examples = [
    {
        "query": "What's the secret to happiness?",
        "answer": "Finding balance in life and learning to enjoy the small moments."
    }, {
        "query": "How can I become more productive?",
        "answer": "Try prioritizing tasks, setting goals, and maintaining a healthy work-life balance."
    }
]

example_template = """
User: {query}
AI: {answer}
"""

example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
)

prefix = """The following are excerpts from conversations with an AI
life coach. The assistant provides insightful and practical advice to the users' questions. Here are some
examples: 
"""

suffix = """
User: {query}
AI: """

few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n\n"
)

# Create the LLMChain for the few-shot prompt template
chain = LLMChain(llm=llm, prompt=few_shot_prompt_template)

# Define the user query
user_query = "What are some tips for improving communication skills?"

# Run the LLMChain for the user query
response = chain.run({"query": user_query})

print("User Query:", user_query)
print("AI Response:", response)