import tiktoken
import openai

class AutoDescription:
    def __init__(self, openai_key, context):
        self.context = context
        # self.initial_description, self.description_PQ, self.potential_query = self.generate_description(self.context)
        openai.api_key = (openai_key)

    def get_context(self):
        return self.context

    def get_initial_description(self):
        return self.initial_description
    
    def get_description_PQ(self):
        return self.description_PQ
    
    def get_potential_query(self):
        return self.potential_query
            
    def num_tokens_from_string(self, string, encoding_name="gpt-3.5-turbo"):
        encoding = tiktoken.encoding_for_model(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens
    
    def generate_description_with_profiler(self, context, model="gpt-3.5-turbo"):
        print("Generating description with profiler...")
        sample_profiler = self.generate_description_initial(context, model)
        print("Generating description with template...")
        template = self.generate_description_template(context, model)
        return sample_profiler, template
    
    def generate_description_with_samples(self, context, model="gpt-3.5-turbo"):
        print("Generating description with samples...")
        sample = self.generate_description_samples(context, model)
        return sample
        
    # def generate_description(self, context, model="gpt-3.5-turbo"):
    #     initial_description_content = self.generate_description_initial(context, model)
    #     potential_query_content = self.generate_potential_query(context, model)
    #     description_PQ_content = self.generate_description_potential_query(initial_description_content, potential_query_content, model)
    #     return initial_description_content, description_PQ_content, potential_query_content

    def generate_description_initial(self, context, model):
        description = openai.ChatCompletion.create(
            model=model,
            messages=[
                    {
                        "role": "system", 
                        "content": "You are an assistant for a dataset search engine.\
                                    Your goal is to increase the performance of this dataset search engine for keyword queries."},
                    {
                        "role": "user", 
                        "content": """Instruction:
Answer the questions while using the input and context.
The input includes dataset title, headers, a random sample, and profiler result of the large dataset.

Input:
""" + context + """
Question:
Describe the dataset covering the nine aspects above in one complete and coherent paragraph.

Answer: """},
                ],
            temperature=0.3)
        description_content = description.choices[0]['message']['content']
        return description_content
    
    def generate_description_samples(self, context, model):
        description = openai.ChatCompletion.create(
            model=model,
            messages=[
                    {
                        "role": "system", 
                        "content": "You are an assistant for a dataset search engine.\
                                    Your goal is to increase the performance of this dataset search engine for keyword queries."},
                    {
                        "role": "user", 
                        "content": """Instruction:
Answer the questions while using the input and context.
The input includes dataset title, headers and a random sample of the large dataset.

Input:
""" + context + """
Question:
Describe the dataset covering the nine aspects above in one complete and coherent paragraph.

Answer: """},
                ],
            temperature=0.3)
        description_content = description.choices[0]['message']['content']
        return description_content
    
    def generate_description_template(self, context, model):
        description = openai.ChatCompletion.create(
            model=model,
            messages=[
                    {
                        "role": "system", 
                        "content": "You are an assistant for a dataset search engine.\
                                    Your goal is to increase the performance of this dataset search engine for keyword queries."},
                    {
                        "role": "user", 
                        "content": """Instruction:
Answer the questions while using the input and context.
The input includes dataset title, headers, a random sample, and profiler result of the large dataset. 

Input:
""" + context + """
Question:
Considering the following nine aspects:
1. Summarize the dataset's overall content and purpose in one sentence.
2. Offer an overview of its structure, including data types and general organization.
3. Identify and group the dataset's headers, explaining their relevance and interrelationships.
4. Detail the value types and range for key headers, emphasizing significant trends or patterns.
5. Describe how the dataset represents time, including the format and temporal scope.
6. Explain the representation of location in the dataset, noting specific formats or geospatial details.
7. Discuss any ambiguities or quality concerns in the data, providing examples where necessary.
8. Highlight any notable findings or potential areas for deeper analysis, based on your initial review.
Describe the dataset covering the nine aspects above in one complete and coherent paragraph.

Answer: """},
                ],
            temperature=0.3)
        description_content = description.choices[0]['message']['content']
        return description_content
    
    

    # def generate_potential_query(self, context, model="gpt-3.5-turbo"):
    #     description = openai.ChatCompletion.create(
    #         model=model,
    #         messages=[
    #                 {
    #                     "role": "system", 
    #                     "content": "You are an assistant for an online dataset search engine.\
    #                                 Your goal is to increase the performance of this dataset search engine for keyword queries."},
    #                 {
    #                     "role": "user", 
    #                     "content": "Answer the questions while using the input and context.\
    #                                 The input is a random sample of the large dataset. \
    #                                 The context describes what the headers of the dataset are.\n"+context+"\
    #                             Question: What can this dataset be used for? Please output top 3 keyword queries where users want to find this data\nAnswer:"},
    #             ],
    #         temperature=0.6)
    #     description_content = description.choices[0]['message']['content']
    #     return description_content

    # def generate_description_potential_query(self, initial_description_content, potential_query_content, model="gpt-3.5-turbo"):
    #     description = openai.ChatCompletion.create(
    #         model=model,
    #         messages=[
    #                 {
    #                     "role": "system", 
    #                     "content": "You are an assistant for an online dataset search engine.\
    #                                 Your goal is to increase the performance of this dataset search engine for keyword queries."},
    #                 {
    #                     "role": "user", 
    #                     "content": "Answer the questions while using the following information.\
    #                                 The description is about a dataset: "+initial_description_content+"\
    #                                 The suggested queries are top keyword queries that users want to find this dataset.\n"+potential_query_content+"\
    #                             Question: Based on the suggested keyword queries. Can you improve the current dataset description to cover these keyword queries? \
    #                                     Only new dataset description is needed in the response.\nAnswer:"},
    #             ],
    #         temperature=0.9)
    #     description_content = description.choices[0]['message']['content']
    #     return description_content
    
    # def generate_description_combined(self, original_description, description_PQ_content, model="gpt-3.5-turbo", temperature=0.9):
    #     description = openai.ChatCompletion.create(
    #         model=model,
    #         messages=[
    #                 {
    #                     "role": "system", 
    #                     "content": "You are an assistant for an online dataset search engine.\
    #                                 Your goal is to increase the performance of this dataset search engine for keyword queries."},
    #                 {
    #                     "role": "user", 
    #                     "content": "Answer the questions while using the following information.\
    #                                 This is the original description about this dataset: "+original_description+"\
    #                                 This is a description based on the suggested keyword queries on this dataset.\n"+description_PQ_content+"\
    #                             Question: Based on these two descriptions. Can you generate a new description by summarizing and highlighting the important parts of these two versions of descriptions? \
    #                                     Only return the new description.\nAnswer:"},
    #             ],
    #         temperature=temperature)
    #     description_content = description.choices[0]['message']['content']
    #     return description_content