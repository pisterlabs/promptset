import numpy as np
import openai

# set api_key
openai.api_key = "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"


class LLMAgent(object):

    def __init__(self):
        self.llm = None
        self.key = None
        self.model = None


    def create_openai_agent(self):
        self.llm = "openai"
        self.key = "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"
        self.model = "gpt-3.5-turbo"


def base_summary(text, ):
    """Generate an initial entity-sparse summary of the given text.

    Args:
    - text (str): The input text to be summarized.

    Returns:
    - str: An entity-sparse summary.
    """

    # initialize an LLM agent using "gpt-3.5-turbo"
    LLM = LLMAgent()
    LLM.create_openai_agent()

    # a step by step prompt
    prompt = ("Given the Text below, you will generate a summary that contains as few entities as possible. Entities are key elements that represent a noun, "
              + "e.g., names of people, companies, brands, cities, countries, etc."
              + "\n Follow these rules below in your response:"
              + "\n1. Minimize Entities: Include as FEW entities as possible in the summary. ONLY include entities that are absolutely necessary and important to the text "
              + "(delete entities that have small impact on the storyline). "
              + "\n2. Use extra wordy phrases: Add a lot of unnecessary connectives in between sentences (e.g., in addition to, on the other hand), modifiers"
              + "(e.g., really good, carefully inspected), fillers (e.g., there is actually)"
              + "\n3. Length: Half of the original text length (e.g. Text length: 20, Summary length: 10)" + "\n\nText: " + text)

    # create a ChatCompletion object to get the reponse of the model
    response = openai.ChatCompletion.create(
        model=LLM.model,
        messages=[
            {"role": "system", "content": "You are an assistant that generates non-specific summary from a given text following user-specified rules."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    # returns the model's output message
    return response.choices[0].message.content


def extract_entities(text):
    """Extract and rank entities from the given text.

    Args:
    - text (str): The input text from which entities should be extracted.

    Returns:
    - list: A list of extracted entities ranked by importance.
    """

    # initialize an LLM agent using "gpt-3.5-turbo"
    LLM = LLMAgent()
    LLM.create_openai_agent()

    # a step by step prompt
    prompt = ("You will extract all the entities from the Text below and rank them by importance. Entities are key elements or information that "
              + "represents a noun, e.g., names of people, companies, brands, cities, countries, etc."
              + "\nFollow these rules below:"
              + "\n1. Extract all entities. If the same entity appears twice, only include it one time in your response."
              + "\n2. Entities of higher importance have bigger impact on the storyline, e.g. protagonist. Entities of lower importance contribute less to the theme of the story"
              + "\n3. Entities should be exactly how they appear in the text, maintaining their original form and spelling."
              + "\n\nResponse: Your response will be a list of comma separated entities. Include all unique entities from the text."
              + "The first entity should be of the highest importance, the second entity should be of the second highest importance, and so on. "
              + "\n\nText: " + text)

    # create a ChatCompletion object to get the reponse of the model
    response = openai.ChatCompletion.create(
        model=LLM.model,
        messages=[
            {"role": "system", "content": "You are a helpful agent that extracts and ranks entities from a given text following user-specified rules."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    # gets the response from the output and transform into list of entities
    return response.choices[0].message.content.strip().split(', ')


def increase_density(summary, entities, target_length):
    """Iteratively incorporate missing entities into the summary without
    increasing its length, using abstraction, fusion, and compression techniques.

    Args:
    - summary (str): The initial summary.
    - entities (list): List of entities to be incorporated into the summary.
    - target_length (int): The desired length of the final summary.

    Returns:
    - str: The final summary with increased density.
    """
    # initialize an LLM agent using "gpt-3.5-turbo"
    LLM = LLMAgent()
    LLM.create_openai_agent()
    prompt = ("Given this summary: " + summary + "\nYour task is to add at least one more entity, ideally 30 percent more entities from: " + str(entities)
              + " to the given summary."
              + "\nFollow these instructions in your response: "
              + "\n1. Word count in the new summary after adding new entities should be EXACTLY " +
              str(target_length)
              + "\n2. Add 30 percent of the given entities that are not already in the given summary."
              + "\n3. Reword the given summary using abstraction, fusion, and compression techniques to include new entities but don't drop any existing entities")

    # create a ChatCompletion object to get the reponse of the model
    response = openai.ChatCompletion.create(
        model=LLM.model,
        messages=[
            {"role": "system", "content": "You are a helpful agent that adds more entities to a given summary without changing the word count of summary"},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content


def evaluate_summaries(summaries):
    """Compare the given summaries and possibly human-written summaries.

    Args:
    - summaries (list): A list of summaries to be evaluated.

    Returns:
    - dict: A dictionary containing evaluation metrics for each summary.
    """

    evaluations = {}

    # iterates through the summaries and evalute the number of entities
    for i in range(len(summaries)):
        # for clarity of the final output, rename the dictionary key
        key_name = "Summary " + str(i+1)
        evaluations[key_name] = len(extract_entities(summaries[i]))

    return evaluations


"""Alternative evaluate_summaries(summaries): We evaluate summaries by the percentage of important entities included.
We can start off by calling extract_entities(text), which will extract and rank all the entities from the original text.
Then we will extract all the entities in each summary and determine the percentage of top-ranked entities present in each summary.
Normally, a better summary will contain more top-ranked entities."""

""" overall code logic - pseducode 
def evaluate_summaries(summaries, entities):
    # Args:
    # - summaries (list): A list of summaries to be evaluated.
    # - entities (list): A list of entities from the original text ranked by importance

    # Returns:
    # - dict: A dictionary containing evaluation metrics for each summary.
    
    evalutaions = {}

    # say top 50 percent entities are important
    topEntities = entities[0:len(entities)/2]

    for summary in summaries: 
        numEntities = extract_entities(summary)
        top = numEntities in topEntities
        evalutaions[summary] = len(top) / len(topEntities)

    return evaluations

"""


def dense_summary(summary, entities, target_length):
    """ incorporate missing entities into the summary without
    increasing its length, using abstraction, fusion, and compression techniques.

    Args:
    - summary (str): The initial summary.
    - entities (list): List of entities to be incorporated into the summary.
    - target_length (int): The desired length of the final summary.

    Returns:
    - str: The final summary with increased density.
    """
    # initialize an LLM agent using "gpt-3.5-turbo"
    LLM = LLMAgent()
    LLM.create_openai_agent()
    prompt = ("Given this summary: " + summary + "\nYour task is to add all of the entities from: " + str(entities)
              + " to this given summary. The new summary should be really entity-dense."
              + "\nFollow these instructions in your response: "
              + "\n1. Word count in the new summary after adding new entities should be EXACTLY " +
              str(target_length)
              + "\n2. Add all of the entities that are not already in the given summary."
              + "\n3. Reword the given summary using abstraction, fusion, and compression techniques to include new entities but don't drop any existing entities")

    # create a ChatCompletion object to get the reponse of the model
    response = openai.ChatCompletion.create(
        model=LLM.model,
        messages=[
            {"role": "system", "content": "You are a helpful agent that adds all given entities to a given summary without changing the word count of summary"},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content


# Example usage:
if __name__ == "__main__":
    text = "Fighter ace: Douglas Bader's story is to be told in a Hollywood film dubbed ‘The First Great Escape’ The story of how Douglas Bader recovered from having both legs amputated to become one of Britain’s greatest fighter aces is remarkable enough. But now another astonishing but little-known chapter of his war career is to be told in a Hollywood film being dubbed ‘The First Great Escape’. Bader – who lost his legs in a flying accident in 1931 – was shot down over France in August 1941 and ended up in a German prison camp in Warburg. It was there that the officer, who always made light of his disability and refused to even use a stick, was involved in a mass break-out that pre-dated the break in 1944 immortalised in The Great Escape starring Steve McQueen and Richard Attenborough. Bader’s life story has already been the subject of the successful film Reach for the Sky in 1956 starring Kenneth More. This latest look at his exploits is based on a book on the Warburg escape by historian Mark Felton called Zero Night. It was a plot hatched by Scottish lieutenant Jock Hamilton-Baillie, 23, and involved build folding ladders to escape over the wire. Major Tom Stallard, of the Durham Light Infantry, teamed up with Bader on the planning. The fighter ace went on to describe it as ‘the most brilliant escape conception of this war’. The ladders were made from wood from a wrecked hut and crafted in the camp’s music room, where the sawing and hammering was drowned out by the sound of instruments. They were disguised as bookshelves. On the night of the break – August 30, 1942 – the prisoners managed to fuse the camp’s search lights and 41 men carrying four 12ft scaling ladders rushed the fence. One ladder collapsed, so only 28 made it over the wire, of which three made a ‘home run’ back to freedom. The Great Escape: Bader's story predates the break in 1944 immortalised in the film starring Steve McQueen and Richard Attenborough (pictured) Bader was among those recaptured but was such a nuisance to the Germans that he ended up in the ‘escape-proof’ Colditz Castle and remained in captivity until the end of the war. More than 40 Allied prisoners put their lives on the line in a plot to escape from Oflag VI-B camp near Warburg, Germany, in 1942. Major Tom Stallard, a larger than life 37-year-old from Somerset captured while serving in the Durham Light Infantry, teamed up with Bader on the planning. While the Great Escape relied on its famous tunnels, the Warburg mass break out saw the men boldly leap over the huge perimeter fences using wooden ladders. Bader described what happened as 'the most brilliant escape conception of this war'. Months of meticulous planning and secret training went into the three minute charge of the camp's double perimeter fences. Bader’s life story has been the subject of 1956 film Reach for the Sky starring Kenneth More (pictured) A series of makeshift ladders propped against the prison camp's perimeter fence were made from wood plundered from a wrecked hut. They were crafted in the camp's music room, where the sawing and hammering was drowned out by the sound of instruments. The escape was codenamed Operation Olympian because it involved troops from across the Commonwealth - Britain, Australia, New Zealand and South Africa. The rights to Dr Felton’s book (pictured) have been bought by production firm Essential 11 USA . As the night of the breakout loomed, the ladders were disguised as bookshelves to fool the guards. After the prisoners fused the perimeter search lights, 41 of them carrying four 12-foot scaling ladders made from bed slats rushed to the barbed-wire fence and clambered over. One ladder collapsed, so only 28 made it over the barbed wire, of which three made a 'home run' back to freedom. Bader later ended up in Colditz and had his tin legs taken away to ensure he remained in captivity until the end of the war. Major Stallard and another leading light of the escape Major Johnnie Cousens, also of the Durham Light Infantry, survived the war and lived into their 70s. Both were too modest to breathe hardly a word about what had happened on August 30 1942. But now their story could trump the Great Escape after Hollywood bosses snapped up the rights to turn military historian Mark Felton's book about the escape Zero Night into a blockbuster. The rights to Dr Felton’s book have been bought by the makers of 2013’s Saving Mr Banks with Tom Hanks. Production firm Essential 11 USA, the makers of the 2013 Tom Hanks and Emma Thompson hit Saving Mr Banks, is now working on a script. Dr Felton said a number of 'A-list' Hollywood stars were queuing up for roles and Essential 11 had invited him to help cast the leads, although he could not reveal who was interested. He said: 'I'm very, very pleased. It's very, very exciting. Essential 11 did a fantastic job with Saving Mr Banks and to have that same team working on this is very pleasing."
    initial_summary = base_summary(text)
    entities = extract_entities(text)
    final_summary = increase_density(initial_summary, entities,
                                     len(initial_summary))
    final_summary_alt = dense_summary(initial_summary, entities,
                                      len(initial_summary))
    summaries = [initial_summary, final_summary, final_summary_alt]
    results = evaluate_summaries(summaries)
    print(results)
