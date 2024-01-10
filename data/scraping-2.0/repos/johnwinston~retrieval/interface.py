from openai import OpenAI
import os

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    return dot_product / (norm_vec1 * norm_vec2)

def most_similar_vector(target_vec, vector_set):
    max_similarity = -1
    most_similar = None

    for vec in vector_set:
        similarity = cosine_similarity(target_vec, vec)
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar = vec

    return most_similar, max_similarity

class Interface:
    def __init__(self):
        self.client = OpenAI()
        self.prompt = Prompts()

    def query_chatGPT(self, message):
        try:
            response = self.client.chat.completions.create(
                messages=message,
                model="gpt-4"
                )
            response = response.choices[0].message.content
        except Exception as e:
            print(e)
            response = ""
        return response

    def get_embedding(self, text):
        return self.client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=text,
                    encoding_format="float"
                    ).data[0].embedding

    def get_description(self, description):
        self.prompt.set_description_prompt()
        self.prompt.prompt_template =\
            self.prompt.prompt_template.format(
                    self.prompt.task_template,
                    description
                    )
        self.prompt.user_content["content"] = (
            self.prompt.user_content["content"].format(
                self.prompt.prompt_template
                )
            )
        self.prompt.messages.append(self.prompt.user_content)
        response = self.query_chatGPT(self.prompt.messages)
        print(response)
        return response.strip()

    def get_descriptions(self,
                        dataset,
                        retrieved_dataset,
                        query
                        ):
        prompt = self.prompt.generate_message(
                    dataset,
                    retrieved_dataset,
                    query
                    )
        response = self.query_chatGPT(prompt)
        orig_desc = response[
                response.find("Original:")+len("Original:")
                :response.find("Retrieved:")
                ]
        ret_desc = response[response.find("Retrieved:")+len("Retrieved:"):]
        return orig_desc.strip(), ret_desc.strip()

class Prompts:
    def __init__(self):
        self.prompt_template = (
            "Task Information:\n{}\n\n"
            "Original Dataset:\n{}\n\n"
            "Retrieved Dataset:\n{}\n\n"
            "Query:\n{}\n\n"
            "Descriptions:\n"
        )
        self.task_template = (
            "We have a system that retrieves datasets based on user queries. The system compares the query with descriptions of original and retrieved datasets. Our goal is to modify these descriptions to ensure the correct dataset is retrieved and irrelevant ones are filtered out.\n"
            "Format the response like this Original: description\nRetrieved: description\n"
            )
        self.messages = [
                {
                    "role" : "system",
                    "content" : "You produce descriptions."
                }
            ]
        self.user_content = {
                "role" : "user",
                "content" : "{}"
                }

    def set_description_prompt(self):
        self.prompt_template = (
            "Task Information:\n{}\n\n"
            "Original Description:\n{}\n\n"
            "New Description:\n"
            )
        
        self.task_template = (
                "Create a dataset of concise, clear, and semantically rich sentence descriptions about various everyday objects, activities, and concepts. Each sentence should be unique and provide specific details that distinguish the subject matter from others in a similar category. Avoid ambiguity and overly complex structures. The descriptions should be suitable for generating embeddings that can be effectively clustered, revealing the nuanced differences and similarities between the subjects. Focus on including relevant keywords and context that capture the essence of each subject, ensuring that the content is diverse yet consistently structured for optimal embedding and clustering performance.\n\n"
                )
        self.messages = [
                {
                    "role" : "system",
                    "content" : "You produce descriptions."
                }
            ]
        self.user_content = {
                "role" : "user",
                "content" : "{}"
                }

    def reset_messages(self):
        self.messages = [
                {
                    "role" : "system",
                    "content" : "You produce descriptions."
                }
            ]

    def format_prompt(self,
                      dataset,
                      retrieved_dataset,
                      query):
        return self.prompt_template.format(
                self.task_template,
                dataset,
                retrieved_dataset,
                query
                )

    def format_user_content(self,
                            dataset,
                            retrieved_dataset,
                            query):
        content = self.user_content.copy()
        content["content"] = self.user_content["content"].format(
                self.format_prompt(
                    dataset,
                    retrieved_dataset,
                    query
                    )
                )
        return content
    def generate_message(self,
                         dataset,
                         retrieved_dataset,
                         query):
        self.messages.append(
            self.format_user_content(
                dataset,
                retrieved_dataset,
                query
                ))
        return self.messages
