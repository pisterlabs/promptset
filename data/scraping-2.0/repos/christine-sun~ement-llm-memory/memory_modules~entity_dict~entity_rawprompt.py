## ENTITY EXTRACTION APPROACH 2:
## Extract entities using GPT prompt and store raw sentences
## about those entities
## Benefits: better than library because you can customize
## prompt. So you can specify that you want the entities
## to be the speakers themselves.
## Downsides: Entity extraction might have errors with GPT
## since the library uses other state-of-the-art techniques

from memory import Memory
import openai
import spacy
from utils import load
import json

class EntityRawPromptMemory(Memory):
    def __init__(self, source_text, k=5):
        super().__init__(source_text)
        entity_dict = {}
        curr_lines = ""
        i = 0

        # Fetch all relevant entities from every line
        # and add them to entity_dict
        for line in source_text.splitlines():
            if i < len(source_text) and i < k:
                curr_lines += line
                curr_lines += "\n"
                i += 1

            # Fetch all entities in curr_lines
            if i == k or i == len(source_text) - 1:

                prompt = f"""
                    Please provide a JSON with the format
                    {{ "entity1": "sentences1", "entity2": "sentences2" }}
                    of all relevant entities mentioned in the following conversation, along with all of their associated sentences, formatted as a JSON object. Make sure the last entry does NOT have a comma after it. Please include the speakers in the conversation as their own entities, and if they have personal information about themselves put all of those as the associated sentences. \n {curr_lines}
                """
                response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=prompt,
                    temperature=0,
                    max_tokens=3000,
                    stop=None,
                    timeout=10
                )
                json_string = response.choices[0].text.replace('[', '').replace(']', '').strip()
                print("This was what i got back")
                print(json_string)
                if json_string[-1] != '}':
                    json_string += '}'
                if json_string[0] != '{':
                    json_string = '{' + json_string
                # Find the last occurrence of the closing curly brace
                last_brace_index = json_string.rfind("}")

                # Find the index of the first non-whitespace character before the closing curly brace
                index = last_brace_index - 1
                while index >= 0 and json_string[index].isspace():
                    index -= 1

                # Check if the character immediately before the closing curly brace is a comma
                if index >= 0 and json_string[index] == ",":
                    json_string = json_string[:index] + json_string[index+1:last_brace_index] + json_string[last_brace_index+1:]
                if json_string[-1] != '}':
                    json_string += '}'

                json_data = json.loads(json_string)
                for key, value in json_data.items():
                    uppercase_key = key.upper()
                    combined_value = ''.join(value)
                    if uppercase_key in entity_dict:
                        entity_dict[uppercase_key] += " " + combined_value
                    else:
                        entity_dict[uppercase_key] = combined_value
                curr_lines = ""
                i = 0

        self.entity_dict = entity_dict
        print("Entity dicgt")
        print(entity_dict)

    def query(self, query):
        # Still use spacy in querying extract entities
        nlp = spacy.load("en_core_web_sm")
        context = ""

        # Find all the relevant entities in the query
        doc = nlp(query)
        for ent in doc.ents:
            print("This is an ent")
            print(ent.text)
            # Put existing entities in context
            if ent.text.upper() in self.entity_dict:
                context += self.entity_dict[ent.text.upper()] + "."
                print("After adding context looks like")
                print(context)

        # Use context to fetch GPT response
        prompt = f"""You are a smart, knowledgable, accurate AI with the following information:
            {context}
        \nYou are sure about your answers. Please answer the following question: {query}
        """
        print("This was the prompt in entityrawprompt")
        print(prompt)
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            temperature=0,
            max_tokens=250,
        )
        return response.choices[0].text.strip()

if __name__ == "__main__":
    source_text = load("test.txt")
    memory_test = EntityRawPromptMemory(source_text)

    query ="What is Mimi's favorite physical activity?"
    answer = memory_test.query(query)
    print(query)
    print(answer)