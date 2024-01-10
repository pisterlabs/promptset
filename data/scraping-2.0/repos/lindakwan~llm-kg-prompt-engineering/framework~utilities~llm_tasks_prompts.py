import openai
import re
import ast
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)


def generate_response(question):
    """
    Generate a response to a question with elaboration.
    :param question: The question to generate a response to.
    :return: The response to the question with elaboration.
    """
    response_json = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You will be provided with a question. Your task is to generate a response to the question."
            },
            {
                "role": "user",
                "content": question
            }
        ],
        temperature=0,
        max_tokens=1000
    )

    return response_json["choices"][0]["message"]["content"]


def generate_response_weaker(question):
    """
    Generate a response to a question using the babbage model.
    :param question: The question to generate a response to.
    :return: The response to the question with elaboration.
    """
    response_json = openai.Completion.create(
        engine="text-babbage-001",
        prompt=question,
        temperature=0,
        max_tokens=1024
    )

    return response_json["choices"][0]["text"]


def generate_response_with_elaboration(question):
    """
    Generate a response to a question with elaboration.
    :param question: The question to generate a response to.
    :return: The response to the question with elaboration.
    """
    elaboration_json = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You will be provided with a question. Your task is to generate a response to the \
                question including elaboration."
            },
            {
                "role": "user",
                "content": question
            }
        ],
        temperature=0,
        max_tokens=1000
    )

    return elaboration_json["choices"][0]["message"]["content"]


def generate_response_using_context_with_elaboration(question, context_string):
    """
    Generate a response to a question with elaboration with context provided.
    :param question: The question to generate a response to.
    :param context_string: The context represented as a string.
    :return: The response to the question with elaboration.
    """
    elaboration_json = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You will be provided with a question and some context. Your task is to generate a \
                response to answer the question including elaboration."
            },
            {
                "role": "user",
                "content": f"Context: {context_string}\nQuestion: {question}"
            }
        ],
        temperature=0,
        max_tokens=1000
    )

    return elaboration_json["choices"][0]["message"]["content"]


def generate_response_using_context(question, context_string):
    """
    Generate a response to a question with context provided.
    :param question: The question to generate a response to.
    :param context_string: The context represented as a string.
    :return: The response to the question with elaboration.
    """
    response_json = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You will be provided with a question and some context. Your task is to generate a \
                response to answer the question."
            },
            {
                "role": "user",
                "content": f"For context: {context_string}\nThe question is: {question}"
            }
        ],
        temperature=0,
        max_tokens=1000
    )

    return response_json["choices"][0]["message"]["content"]


def generate_response_using_context_weaker(question, context_string):
    """
    Generate a response to the question provided the context using the babbage model.
    :param question: The question to feed into the model.
    :param context_string: The string containing the context.
    :return: The response to the question.
    """
    response_json = openai.Completion.create(
        engine="text-babbage-001",
        prompt=f"Here is the context: {context_string}\nThe question is: {question}",
        temperature=0,
        max_tokens=1024
    )

    return response_json["choices"][0]["text"]


def extract_entities(text):
    """
    Extract the entity names from the text.
    :param text: The text to extract entities from.
    :return: The list of entity names extracted from the text.
    """
    entities_json = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You will be provided with text. \
                    Your task is to identify a list of entity names mentioned in the text. No documentation, \
                    no explanation, only python3 list code, escape all apostrophes with backslash."
            },
            {
                "role": "user",
                "content": f"Text: {text}"
            }
        ],
        temperature=0,
        max_tokens=256
    )
    extracted_entities = entities_json["choices"][0]["message"]["content"]
    init_entity_names = ast.literal_eval(extracted_entities)

    entity_names = []

    # Split entities with "and" into two parts
    for ent_name in init_entity_names:
        if "and" in ent_name:
            split_names = " and ".split(ent_name)
            entity_names.extend(split_names)
        else:
            entity_names.append(ent_name)

    return entity_names


def extract_relations(text):
    relations_json = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You will be provided with text. \
                    Your task is to identify a list of predicate names mentioned in the text. No documentation, \
                    no explanation, only python3 list code, escape all apostrophes with backslash."
            },
            {
                "role": "user",
                "content": f"Text: {text}"
            }
        ],
        temperature=0,
        max_tokens=256
    )
    extracted_relations = relations_json["choices"][0]["message"]["content"]
    relation_names = ast.literal_eval(extracted_relations)
    return relation_names


def extract_kg_facts(text, entities, relations):
    """
    Extract the triples from the text given the lists of entities and relations.
    :param text: The text to extract triples from.
    :param entities: The list of entities to choose from.
    :param relations: The list of relations to choose from.
    :return: The list of triples extracted from the text.
    """
    llm_facts_json = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You will be provided with text, list of entities, and list of relations. \
                    Using the lists of entities and relations, your task is to extract triples from the text in \
                    the form (subject, predicate, object)"  # (subject URI, predicate URI, object URI)."
            },
            {
                "role": "user",
                "content": f"Text: {text}\nEntities: {entities}\nRelations: {relations}"
            }
        ],
        temperature=0,
        max_tokens=256
    )

    triples = re.findall(r"\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*\)",
                         llm_facts_json["choices"][0]["message"]["content"], re.IGNORECASE)

    return triples


def extract_kg_facts_given_entities(text, entities):
    """
    Extract the triples from the text given the lists of entities.
    :param text: The text to extract triples from.
    :param entities: The list of entities to choose from.
    :return: The list of triples extracted from the text.
    """
    llm_facts_json = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You will be provided with text and a list of entities. \
                    Using the lists of entities, your task is to extract triples from the text in \
                    the form (subject, predicate, object)."
            },
            {
                "role": "user",
                "content": f"Text: {text}\nEntities: {entities}"
            }
        ],
        temperature=0,
        max_tokens=256
    )

    triples = re.findall(r"\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*\)",
                         llm_facts_json["choices"][0]["message"]["content"], re.IGNORECASE)

    return triples


def get_similar_identifier_given_context(item_name, context, item_type="property"):
    similar_identifier_json = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "Just provide the identifier, no other explanation or documentation."
            },
            {
                "role": "user",
                "content": f'What is the most similar Wikidata URI for the {item_type} "{item_name}" in context \
                of sentence "{context}"?'
            }
        ],
        temperature=0,
        max_tokens=256
    )

    opt = similar_identifier_json["choices"][0]["message"]["content"]

    print("Id output:", opt)

    if "P" in opt:
        ids = re.findall(r'P\d+', opt)
        if len(ids) == 0:
            ids2 = re.findall(r'\d+', opt)
            identifier = "wdt:P" + re.findall(r'\d+', opt)[0]
        else:
            identifier = "wdt:" + ids[0]
    elif "Q" in opt:
        ids = re.findall(r'Q\d+', opt)
        if len(ids) == 0:
            ids2 = re.findall(r'\d+', opt)
            identifier = "wd:Q" + re.findall(r'\d+', opt)[0]
        else:
            identifier = "wd:" + ids[0]
    elif item_type == "item":
        identifier = "wd:Q" + re.findall(r'\d+', opt)[0]
    else:
        identifier = "wdt:P" + re.findall(r'\d+', opt)[0]

    return identifier


def extract_relevant_predicates(text, predicates, k=3):
    """
    Extract the top k most relevant predicates to the text.
    :param text: The text to extract relevant predicates from.
    :param predicates: The list of predicates to choose from.
    :param k: The number of predicates to return.
    :return: The top k most relevant predicates to the text.
    """
    relevant_preds_json = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": f"You will be provided with question text and a list of predicates. \
                Your task is to order the {k} most relevant predicates to the question by most relevant to question \
                which would be the most helpful in answering the question. \
                No documentation, no explanation, only valid python3 list code, escape all apostrophes with backslash."
            },
            {
                "role": "user",
                "content": f"Text: {text}\nPredicates: {predicates}"
            }
        ],
        temperature=0,
        max_tokens=2000
    )

    relevant_preds_opt = relevant_preds_json["choices"][-1]["message"]["content"]

    # print("Relevant predicates output:", relevant_preds_opt)

    # Escape apostrophes
    escaped_list = re.sub(r"(?<=')([^',\[\]]*)(?<!\\)'([^',\[\]]*)(?=')", r"\1\\'\2", relevant_preds_opt)

    # Replace newlines with spaces
    escaped_list = re.sub(r"\n", " ", escaped_list)

    print("Escaped relevant predicates output:", escaped_list)

    relevant_preds = ast.literal_eval(re.findall(r'\[.*?\]', escaped_list)[0])

    # Get the top k most relevant predicates
    top_preds = relevant_preds[:k]

    return top_preds


def extract_relevant_facts(text, facts, k=5):
    if k == 0:
        return []

    relevant_facts_json = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": f"You will be provided with question text and a list of triples, each triple in the form \
                (subject, predicate, object). \
                Your task is to order the {k} most relevant triples to the question by most relevant to question. \
                No documentation, no explanation, only a single syntactically valid python3 list as code, \
                escape all apostrophes with backslash, no ellipsis at all, no '...' at all."
            },
            {
                "role": "user",
                "content": f"Question Text: {text}\nFacts: {facts}"
            }
        ],
        temperature=0,
        max_tokens=2000
    )

    relevant_facts_opt = relevant_facts_json["choices"][-1]["message"]["content"]

    # print("Relevant facts output:", relevant_facts_opt)

    # Escape apostrophes
    escaped_list = re.sub(r"(?<=')([^',\[\]]*)(?<!\\)'([^',\[\]]*)(?=')", r"\1\\'\2", relevant_facts_opt)

    # Replace newlines with spaces
    escaped_list = re.sub(r"\n", " ", escaped_list)

    # Escape square brackets
    escaped_list = '[' + escaped_list[1:-1].replace('[', r'\[').replace(']', r'\]') + ']'

    # print("Escaped relevant facts output:", escaped_list)

    try:
        relevant_facts = ast.literal_eval(re.findall(r'\[.*?\]', escaped_list)[0])
        # Get the top k most relevant facts
        top_facts = relevant_facts[:k]
        top_facts = list(filter(lambda x: len(x) == 3, top_facts))
        return top_facts
    except:
        # print("Error in extracting relevant facts.")
        return extract_relevant_facts(text, facts, k-1)


def select_mc_response_based(question, response, choices):
    """
    Select the best multiple choice response based on the question and response.
    :param question: The question fed into the model.
    :param response: The response generated by the model.
    :param choices: The multiple choice options.
    :return: The letter output of the best choice.
    """
    mc_prompt = PromptTemplate(
        input_variables=["question", "response", "choices"],
        template="Output the best one of the numbered options (1-4) for the following question and response:\n \
                            Question: {question}\nResponse: {response}\nOptions:\n{choices}"
    )

    choices_text = "\n".join([str(i + 1) + ". " + choice for i, choice in enumerate(choices)])
    choice_response = llm(mc_prompt.format(question=question, response=response, choices=choices_text))

    print("Choice Response:", choice_response.strip())

    # Convert the response to the numbered choice
    numbers = [int(num) for num in re.findall(r'\d+', choice_response.strip().split(".")[0])]
    if len(numbers) == 0:
        # Choose A by default if no output
        numbered_output = 1
    else:
        numbered_output = numbers[-1]
    letter_output = chr(ord('A') + int(numbered_output) - 1)

    # print("Generated answer:", letter_output)

    return letter_output
