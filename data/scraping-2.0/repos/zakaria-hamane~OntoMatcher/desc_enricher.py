from langchain.utilities import WikipediaAPIWrapper
import json


class DescEnricher:
    def __init__(self, lang: str = "en"):
        self.lang = lang
        self.wikipedia = WikipediaAPIWrapper()

    def enrich(self, entity: str) -> str:
        result = self.wikipedia.run(entity)

        summary = ""
        start_index = result.find("Summary: ")
        if start_index != -1:
            start_index += len("Summary: ")
            end_index = result.find("\n\n", start_index)
            if end_index != -1:
                summary = result[start_index:end_index]

        return summary

def run():
    desc_enricher = DescEnricher()
    # for entity in extracted_data/enriched/all_empty_desc.json add key "ent_desc" with the value of the description from Wikipedia
    with open('extracted_data/enriched/all_empty_desc.json') as json_file:
        entities = json.load(json_file)

    updated_entities = []
    for i in range(len(entities)):
        query_dict = entities[i]
        query = list(query_dict.keys())[0]
        desc = desc_enricher.enrich(query)
        print(f"Enriched {query}")
        updated_entities.append({query: desc})  # add the description to the key's value

    with open('extracted_data/enriched/all_empty_desc.json', 'w') as outfile:
        json.dump(updated_entities, outfile, indent=4)


if __name__ == "__main__":
    run()

# Path: desc_enricher.py
