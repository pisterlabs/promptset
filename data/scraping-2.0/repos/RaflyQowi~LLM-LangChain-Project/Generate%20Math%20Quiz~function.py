from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.llms import HuggingFaceHub
from dotenv import load_dotenv

load_dotenv();
config = {'max_new_tokens': 512, 'temperature': 0.6}

# Create function for app
def GetLLMResponse(selected_topic_level, 
                    selected_topic,
                    num_quizzes):
    
    # Calling llama model
    # llm = CTransformers(model="D:\Code Workspace\DL Model\llama-2-7b-chat.ggmlv3.q8_0.bin",
    #                     model_type = 'llama',
    #                     config = config)

    # llm = CTransformers(model='TheBloke/Llama-2-7B-Chat-GGML',
    #                     model_file = 'llama-2-7b-chat.ggmlv3.q8_0.bin',
    #                     model_type = 'llama',
    #                     config = config)

    llm = HuggingFaceHub(
        repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1",
        model_kwargs = config
    )
    
    ## Create LLM Chaining
    questions_template = "I want you to just generate question with this specification: Generate a {selected_topic_level} math quiz on the topic of {selected_topic}. Generate only {num_quizzes} questions not more and without providing answers. The Question should not in image format/link"
    questions_prompt = PromptTemplate(input_variables=["selected_topic_level", "selected_topic", "num_quizzes"],
                                      template=questions_template)
    questions_chain = LLMChain(llm= llm,
                               prompt = questions_prompt,
                               output_key = "questions")


    answer_template = "I want you to become a teacher answer this specific Question:\n {questions}\n\n. You should gave me a straightforward and consise explanation and answer to each one of them"
    answer_prompt = PromptTemplate(input_variables = ["questions"],
                                   template = answer_template)
    answer_chain = LLMChain(llm = llm,
                            prompt = answer_prompt,
                            output_key = "answer")
    
    ## Create Sequential Chaining
    seq_chain = SequentialChain(chains = [questions_chain, answer_chain],
                                input_variables = ['selected_topic_level', 'selected_topic', 'num_quizzes'],
                                output_variables = ['questions', 'answer'])
    
    response = seq_chain({'selected_topic_level': selected_topic_level, 
                            'selected_topic': selected_topic, 
                            'num_quizzes' : num_quizzes})
    
    ## Generate the response from the llama 2 model
    print(response)
    return response