## ENTITY EXTRACTION APPROACH 1:
## Extract entities using spaCy and store raw sentences
## about those entities
## Flaws: using a library makes it difficult to identify
## information related to a speaker. Only stores entities
## talked about within the sentences themselves. This is
## because a library is hard to customize

from memory import Memory
import openai
import spacy
from utils import load

class EntityRawSpaCyMemory(Memory):
    def __init__(self, source_text):
        super().__init__(source_text)
        nlp = spacy.load("en_core_web_sm")
        entity_dict = {}

        # Fetch all relevant entities from every line
        # and add them to entity_dict
        for line in source_text.splitlines():
            doc = nlp(line)
            for ent in doc.ents:
                # If ent exists, append this new line to its
                # raw sentences
                if ent.text.upper() in entity_dict:
                    entity_dict[ent.text.upper()] += " " + ent.sent.text
                else:
                    entity_dict[ent.text.upper()] = ent.sent.text

        self.entity_dict = entity_dict
        print("Entity dict")
        print(entity_dict)

    def query(self, query):
        nlp = spacy.load("en_core_web_sm")
        context = ""

        # Find all the relevant entities in the query
        print("This is my query")
        print(query)
        doc = nlp(query)
        for ent in doc.ents:
            print("This is my entity")
            print(ent.text)
            # Put existing entities in context
            if ent.text.upper() in self.entity_dict:
                context += self.entity_dict[ent.text.upper()] + ". "
        print("Done looking at similar entities")

        # Use context to fetch GPT response
        prompt = f"""You are a smart, knowledgable, accurate AI with the following information:
            {context}
        \nYou are sure about your answers. Please answer the following question: {query}
        """
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            temperature=0,
            max_tokens=250,
        )
        return response.choices[0].text.strip()

if __name__ == "__main__":
    source_text = load("test.txt")
    memory_test = EntityRawSpaCyMemory(source_text)

    query ="What is Mimi's favorite physical activity?"
    answer = memory_test.query(query)
    print(query)
    print(answer)

