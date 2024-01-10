from langchain import PromptTemplate, FewShotPromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
import os
import json
import argparse
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector, MaxMarginalRelevanceExampleSelector

# We need to add the OpenAI API key to the environment variables for using embeddings and llm.
os.environ["OPENAI_API_KEY"] = "YOU KEY HERE"

# Read jsonl file containing LM-KBC data
def read_lm_kbc_jsonl(file_path: str):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def create_property_lookup(data):
    property_lookup = {}
    for item in data['input']:
        property_name = item['property']
        output_info = item['output'][0]
        property_lookup[property_name] = output_info

    return property_lookup

def lookup_property_info(property_lookup, lookup_property):
    if lookup_property in property_lookup:
        property_info = property_lookup[lookup_property]

        value = property_info['value']
        wikidata_id = property_info['wikidata_id']
        wikidata_label = property_info['wikidata_label']
        domain = property_info['domain']
        range_ = property_info['range']
        explanation = property_info['explanation']

        return {
            "Property": lookup_property,
            "Value": value,
            "Wikidata ID": wikidata_id,
            "Wikidata Label": wikidata_label,
            "Domain": domain,
            "Range": range_,
            "Explanation": explanation
        }
    else:
        return f"Property '{lookup_property}' not found in the lookup dictionary."
    
def process_train_data(train_data, property_lookup):
    examples_with_wiki = []

    for line in train_data:
        subject = line['SubjectEntity']
        relation = line['Relation']
        objects = str(line['ObjectEntities'])
        rel_props = lookup_property_info(property_lookup, relation)

        if rel_props == f"Property '{relation}' not found in the lookup dictionary.":
            domain = 'unknown'
            range_value = 'unknown'
            wikidata_label = 'unknown'
            explanation = 'unknown'
        else:
            domain = rel_props['Domain']
            range_value = rel_props['Range']
            wikidata_label = rel_props['Wikidata Label']
            explanation = rel_props['Explanation']

        instance_dict = {
            'entity_1': subject,
            'domain': domain,
            'range': range_value,
            'relation': relation,
            'wikidata_label': wikidata_label,
            'explanation': explanation,
            'target_entities': objects
        }
        examples_with_wiki.append(instance_dict)

    return examples_with_wiki


def compile_prompt(string_template, example_variables, 
                   examples, example_selector_type, k,
                   embeddings, 
                   prefix, suffix, input_variables):
    """
    formatter : string_template with placeholders for the input variables
    
    prompt_template : PromptTemplate object
                        takes formatter and example_variables:list of strings
                        
    example_selector_type : string, 'semantic_similarity' or 'max_marginal_relevance'
    
    k: int, number of examples to produce
                        
    example_selector : SemanticSimilarityExampleSelector, MaxMarginalRelevanceExampleSelector object
                        takes examples (list of dictionaries), embeddings, vector store, k=nr of examples to produce
                        
    embeddings : OpenAI embeddings 
                    takes nothing
                    
    few_shot_template : FewShotPromptTemplate object
                        takes example_selector, 
                            prompt_template, 
                            prefix, suffix, 
                            input_variables, it is different from example_variables because we don't have the targets in the input_variables
                            example_separator
    """	
    formatter= string_template
    prompt_template = PromptTemplate(
                input_variables=example_variables,  
                template=formatter,
                )
    examples = examples
    
    k=k
    
    embeddings = embeddings
    
    if example_selector_type == 'semantic_similarity':
        example_selector = SemanticSimilarityExampleSelector.from_examples(
            examples,
            embeddings,
            Chroma,
            k
            )
    if example_selector_type == 'max_marginal_relevance':
        example_selector = MaxMarginalRelevanceExampleSelector.from_examples(
            examples,
            embeddings,
            Chroma,
            k
            )
        
    few_shot_template = FewShotPromptTemplate(
        example_selector= example_selector, 
        example_prompt=prompt_template,
        prefix=prefix,
        suffix=suffix,
        input_variables=input_variables,
        example_separator="\n"
        )
    
    return few_shot_template


def generate_prompt(subj, rel,
                    input_file='../data/random_val_sample2.jsonl',
                    wiki_properties='../data/relevant_wiki_properties_by_chatgpt.json',
                    formatter="""Given Entity: {entity_1} 
                        Domain of the Given Entity: {domain}  
                        Range of the Given Entity:: {range} 
                        Given Relation: {relation} 
                        Wikidata label of the given relation: {wikidata_label} 
                        Wikidata explanation of the given relation: {explanation} 
                        ==> 
                        Target entities: {target_entities} 
                        """,
                    example_variables=['entity_1', 'domain', 'range', 'relation', 'wikidata_label', 'explanation', 'target_entities'],
                    example_selector_type='semantic_similarity',
                    k=4,
                    prefix="""I would like to use you as a knowledge base. I am going to give you an entity and relation pair. 
              I want you to generate new entities holding the relation with the given entity. Number of answers may vary between 0 to 16. 
              I will show you some examples. Act like a knowledge base and do your best! Here we start. Examples: """,
                    suffix="""End of the examples. Now it is your turn to generate.
                        Given Entity: {entity_1} 
                        Domain of the Given Entity: {domain}  
                        Range of the Given Entity:: {range} 
                        Given Relation: {relation} 
                        Wikidata label of the given relation: {wikidata_label} 
                        Wikidata explanation of the given relation: {explanation} 
                        ==> 
                        Target entities: """,
                    input_variables=['entity_1', 'domain', 'range', 'relation', 'wikidata_label', 'explanation']
                    ):
    # parser = argparse.ArgumentParser(description='Compile prompt for few-shot learning with LangChain example selectors')
    #
    # # parser.add_argument('-subject', '--subject', type=str, default='Coldplay', help='Given subject entity')
    # # parser.add_argument('-relation', '--relation', type=str, default='BandHasMember', help='Given relation')
    # parser.add_argument('-i', '--input_file', type=str, default= '../data/random_val_sample2.jsonl', help='Directory with data to select from')
    # parser.add_argument('-wiki', '--wiki_properties', type=str, default='../data/relevant_wiki_properties_by_chatgpt.json', help='Wikidata properties generated by ChatGPT')
    # parser.add_argument('-formatter', '--formatter', type=str, default= """Given Entity: {entity_1}
    #                     Domain of the Given Entity: {domain}
    #                     Range of the Given Entity:: {range}
    #                     Given Relation: {relation}
    #                     Wikidata label of the given relation: {wikidata_label}
    #                     Wikidata explanation of the given relation: {explanation}
    #                     ==>
    #                     Target entities: {target_entities}
    #                     """, help='String template for the prompt')
    # parser.add_argument('-example_variables', '--example_variables', type=list, default=['entity_1', 'domain', 'range', 'relation', 'wikidata_label', 'explanation', 'target_entities'], help='List of variables in the string template')
    # parser.add_argument('-example_selector_type', '--example_selector_type', type=str, default='semantic_similarity', help='Type of example selector, either "semantic_similarity" or "max_marginal_relevance"')
    # parser.add_argument('-k', '--k', type=int, default=3, help='Number of examples to produce')
    # parser.add_argument('-prefix', '--prefix', type=str, default="""I would like to use you as a knowledge base. I am going to give you an entity and relation pair.
    #           I want you to generate new entities holding the relation with the given entity. Number of answers may vary between 0 to 16.
    #           I will show you some examples. Act like a knowledge base and do your best! Here we start. Examples: """, help='Prefix for the prompt')
    # parser.add_argument('-suffix', '--suffix', type=str, default="""End of the examples. Now it is your turn to generate.
    #                     Given Entity: {entity_1}
    #                     Domain of the Given Entity: {domain}
    #                     Range of the Given Entity:: {range}
    #                     Given Relation: {relation}
    #                     Wikidata label of the given relation: {wikidata_label}
    #                     Wikidata explanation of the given relation: {explanation}
    #                     ==>
    #                     Target entities: """, help='Suffix for the prompt')
    # parser.add_argument('-input_variables', '--input_variables', type=list, default=['entity_1', 'domain', 'range', 'relation', 'wikidata_label', 'explanation'], help='List of variables in the input')
    #
    # args = parser.parse_args()

    train_data = read_lm_kbc_jsonl(input_file)
    print(f'Number of training examples: {len(train_data)}')

    with open(wiki_properties) as f:
        wiki_properties = json.load(f)
  
    property_lookup = create_property_lookup(wiki_properties)

    examples_with_wiki = process_train_data(train_data, property_lookup)

    # calling the embeddings
    openai_embeddings = OpenAIEmbeddings()
    
    few_shot_template = compile_prompt(string_template=formatter,
                                   example_variables= example_variables,
                                    examples= examples_with_wiki, 
                                    example_selector_type = example_selector_type,
                                    k=k,
                                    embeddings= openai_embeddings, 
                                    prefix=prefix,
                                    suffix=suffix,
                                    input_variables=input_variables,
                                    )
    
    ent_1 = subj
    relation = rel
    rel_props = lookup_property_info(property_lookup, relation)
    print(property_lookup)
    print(rel_props)
    
    prompt = few_shot_template.format(entity_1= ent_1, relation=relation, 
                                  domain=rel_props['Domain'], 
                                  range= rel_props['Range'], 
                                    wikidata_label= rel_props['Wikidata Label'], 
                                    explanation= rel_props['Explanation'])
    print('Final test')
    print(prompt)
    
    return prompt
    
    
    
    
if __name__ == '__main__':
    print()
    # main()
