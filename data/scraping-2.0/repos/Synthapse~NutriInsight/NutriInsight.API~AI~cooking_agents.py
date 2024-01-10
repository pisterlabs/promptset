# from langchain import PromptTemplate, LLMChain, Prompt
# from AI.agents import recipe_prompt, recipe_prompt_history
# from langchain.llms import LlamaCpp
# from langchain.callbacks.manager import CallbackManager
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# class LlamaAgent:

#     def __init__(self):
#         self.llm = None
#         self.start_history = "You are an AI assistant created by Cognispace to generate recipes. Your decisions should be made independently without seeking user assistance. GOALS: - Understand the user's desired mood from their input. - Suggest recipes fitting that mood using available ingredients. - Ensure recipes align with any user constraints. CONSTRAINTS: - Ask about allergy and diet restrictions to avoid unsafe recommendations. - If ingredients are limited, suggest reasonable substitutions. - Validate recipes meet all user criteria before suggesting. - Be honest if an appropriate recipe isn't possible. - Offer to try again with more info. IMPORTANTLY, format your responses as JSON with double quotes around keys and values, and commas between objects. "
#         self.history = "You are an AI assistant created by Cognispace to generate recipes. Your decisions should be made independently without seeking user assistance. GOALS: - Understand the user's desired mood from their input. - Suggest recipes fitting that mood using available ingredients. - Ensure recipes align with any user constraints. CONSTRAINTS: - Ask about allergy and diet restrictions to avoid unsafe recommendations. - If ingredients are limited, suggest reasonable substitutions. - Validate recipes meet all user criteria before suggesting. - Be honest if an appropriate recipe isn't possible. - Offer to try again with more info. IMPORTANTLY, format your responses as JSON with double quotes around keys and values, and commas between objects. "


#     def initialize_llama(self):
#         callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
#         self.llm = LlamaCpp(
#             model_path="AI/Llama2/llama7B-2.bin",
#             temperature=0.25,
#             max_tokens=2000,
#             top_p=1,
#             callback_manager=callback_manager,
#             verbose=True,
#             use_mlock=True,
#         )


#     def generate(human_input):

#         print("Loading Llama model...")

#         prompt: PromptTemplate = recipe_prompt

#         chain = LLMChain(
#             llm=llm, 
#             prompt=prompt,
#         )

#         response = chain.predict(user_input=human_input)
#         return response
    
#     def generate_conversations(self, human_input):

#         print ("Start conversation with history...")

#         #  "error": "Requested tokens (521) exceed context window of 512"
#         print(self.history)
#         print(len(self.history))

#         if len(self.history) > 500:
#             self.history[-800:]

#         prompt_filled = PromptTemplate(input_variables = [],template=recipe_prompt_history.format(history=self.history, user_input=human_input))

#         chain = LLMChain(
#             llm=llm, 
#             prompt=prompt_filled,
#         )

#         ai_response = chain.predict(user_input=prompt_filled)
#         self.history += f"\nHuman: {human_input}\nAI: {ai_response}"

#         print(self.history)

#         return self.history[len(self.start_history):]

