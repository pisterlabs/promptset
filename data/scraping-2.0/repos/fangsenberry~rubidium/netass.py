'''
This file contains the stuff for net assessment.

1. Multithreaded approach to net assessment

Generally speaking only one function calls things in this file, which is the net assessment function from the main file.
'''

#my own files
import helpers

#installed libs
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#native libs
import concurrent.futures
import tkinter as tk

from time import sleep

#initialize all keys and relevant information
chosen_model = "gpt-4"

'''
TODO: when we flesh out Ruby with few shot NA examples, there needs to be a rethink as to how the plan should be executed. e.g. each step of the plan should consider few shot databases.
Taking a question, devises a full plan to prepare for answering that question. This is a precursor to our search function, and serves as our information gathering fucntion for now, but will be expanded to include usage of tools in the future for Ruby to use. This includes

@params:
    question [string]: the question that we are trying to answer
    tools [list] : a list of functions and descriptions of them that Ruby can decide to use or not. It also should include descriptions of our local knowledge bases NOT IMPLEMENTED YET

@returns:
    the plan [string] : the plan is returned by GPT in a specified format, which is then parsed by the parse_plan function, which translates all of those plans into actions that can be executed by code. TODO: take a look at the json function calling documentation for GPT to directly call the functions that we need it to call. We will also pass this plan into the large net assessment function, so at each step, our plan for answering the question is considered.

'''
def plan_approach(question, tools=None):
    system_init = f"""You are Rubidium. You are a world-class Geopoltical Analyst, who is extremely good at breaking down and planning comprehensive approaches to tough and complex analysis questions and research areas. You follow the concepts of Net Assessment, and you are the best in the world at Net Assessment.
    
    Here is a reference to what Net Assessment is: Net Assessment is a strategic evaluation framework that carries considerable significance in the field of geopolitical and military analysis. It was pioneered by the Office of Net Assessment (ONA) in the United States Department of Defense in 1973, reflecting its rich historical context, but its utility today is felt beyond the shores of a single nation, offering globally pertinent insights for any entity faced with complex geopolitical dynamics. 

    This methodical process undertakes a comparative review of a range of factors including military capabilities, technological advancements, political developments, and economic conditions among nation-states. The primary aim of Net Assessment is to identify emerging threats and opportunities, essentially laying the groundwork for informed responses to an array of possible scenarios, making it a powerful tool in modern geopolitical and military strategy.

    Net Assessment examines current trends, key competing factors, potential risks, and future prospects in a comparative manner. These comprehensive analyses form the bedrock of strategic predictions extending up to several decades in the future. Thus, leaders geared towards long-term security and strategic outlooks stand to benefit significantly from this indispensable tool.

    The framework also paves the way for diverse types of materials and findings, ranging from deeply researched assessments to concise studies, informal appraisals, and topical memos. These resources, although initially produced in a highly classified environment, have been toned down and adapted for broader strategic and policy-related debates, serving as critical inputs for decision-makers in diverse geopolitical contexts. 

    The role of Net Assessment in shaping historical shifts in policy and strategy merits attention. Despite acknowledging its roots within the US Department Defense, it’s important to note its influence on significant decisions. For instance, it has helped draft new strategic paradigms and reverse major policy decisions, exhibiting its potential for globally-relevant strategic discourse.

    In summary, while Net Assessment's roots trace back to the US defense context, its relevance spreads far wider today. It has emerged as a universally applicable strategic tool in geopolitical and military analysis, enabling nations to navigate intricate dynamics, anticipate future trends, and formulate informed strategies. Its role in shaping policy decisions and strategic discourse underscores its enduring importance in an increasingly complex global context."""

    prompt = f"""
    I will give you a research question, and you will break down and plan how you would approach that question. The research questions are purposefully complex and are of incredible depth, and therefore your plan must have equal depth and complexity. You must approach this question in a recursive way, breaking down the question in the smaller parts until you reach a final, base case where you have all the actions and information needed for you to fully answer this research question. You should be imaginative and consider different and unique perspective that might be able to better help you answer the provided question. At each step, you should elucidate research components that need further research. For example, if you are asking to assess the impact of AI on Japan's economy, then this needs more research as to what Japan's economy contains and is made up by, and you should also search the news for what Japan is planning to do with AI in order to better assess the impacts of AI in Japan. Research actions should be tagged with a [RESEARCH] tag. You must give me the actions IN ORDER, but you must not add any form of numbering. Follow the belowmentioned format exactly.

    Here is the research question: {question}

    Below, I have provided you a sample output format. You MUST follow the output format given to you below.

    First Example Input:
    How would the widespread implementation of AI technologies affect unemployment and job displacement trends across different economies?
    
    First Example Output:
    Approach: I should search for the latest news on AI, and just to be safe, also search for the current foreseeable impacts of AI. I should also retrieve information and unemployment and job displacement rates for the largest economies, so as to get a better understanding of the current situation. I should then look at the components of each large economy, in order to determine what they are comprised of. In order to do so, I should retrieve data on what these economies GDP's are comprised of. I should split these up into seperate search queries in order for me to accurately retrieve information about each one. I should also search for and retrieve a general overview on how the world's economies are structured, and what is the job distribution in first, second, and third world countries, in order for a better overview on everything.

    Actions:
    [RESEARCH] Search for the latest news on AI
    [RESEARCH] Search for the current foreseeable impacts of AI
    Determine which are the largest economies in the world
    [RESEARCH] Determine which are the largest economies that we want to focus on to determine the impact of AI
    [RESEARCH] Retrieve information on unemployment and job displacement rates for the largest economies
    [RESEARCH] Retrieve information on what the largest economies GDP's are comprised of (Split these up into seperate search queries)
    [RESEARCH] Search for a general overview on how the world's economies are structured.
    With all that information, I have enough to come up with the answer.
    """

    try_count = 1
    while try_count <= 10:
        try:
            response = client.chat.completions.create(model=chosen_model,
            messages=[
                {"role": "system", "content": system_init},
                {"role": "user", "content": prompt}
            ],
            temperature=0.9)
            break
        except Exception as e:
            rest_time = 10 * try_count
            print(f"Encountered Error: {e}. Retrying in {rest_time} seconds...")
            sleep(rest_time)
            try_count += 1
            print(f"Retrying summarisation...")


    return response.choices[0].message.content

'''
Takes in a plan by GPT and then extracts the questions that we need to ask in order. Specificied format from our end. These questions are going to go into our search queries. TODO: take a look at the json function calling documentation for GPT to directly call the functions that we need it to call. The plan will largely be information gathering, so we can go ahead and let the execution layer (this function) return the insights that have been found

@params:
    plan [string]: the plan given by GPT

@returns:
    results [string] : a string representing the results from carrying out the plan given.

'''
def parse_plan(plan, question, tools=None):
    print("starting to parse plan")
    system_init = f"""You are ParseGPT. You are an AI that specialises in extracting plans from texts, and formatting them in a specified format."""

    prompt = f"""
    I will give you a plan in ordered steps, as well as the question that this plan was built to answer. This plan includes explanations for why the plan is structured as such. All you have to do is extract the steps from the plan, and then format them in a specified format. You must give me the steps IN ORDER, but you must not add any form of numbering. Follow the belowmentioned format exactly. You MUST only extract the steps. You MUST NOT add any additional steps or explanations, or other kinds of formatting. Your output will be programmatically handled. So there MUST NOT be any other tokens other than the raw text of the steps in the plan.

    Here is the plan: {plan}
    Here is the question: {question}

    Example Input:
    Approach: To answer the research question, first, there should be an analysis of the impact of AI advancements on the creation of new industries and consumption patterns. Then, these impacts should be considered in the context of their possible effects on global economic growth, wealth distribution, and socioeconomic inequalities.

    Actions:
    [RESEARCH] Start by investigating the latest global trends in the AI industry to understand the current state and its forecasted developments.
    [RESEARCH] Look into specifics about how AI technology is expected to innovate traditional industries and create new ones.
    [RESEARCH] Analyze the correlation between the adoption of AI technologies and the transformation of consumption patterns.
    [RESEARCH] Study historical precedents related to technological advancements and their effects on the global economy to provide a context for speculation.

    By now, I should have a clear understanding of how AI advancements are shaping industries and consumption patterns. The next part is to estimate the potential impacts on global economic growth, wealth distribution, and socioeconomic inequalities.

    [RESEARCH] Examine the current global economic growth rates and compare them across countries while focusing on the role technology plays.
    [RESEARCH] Investigate how the wealth is distributed globally (across countries, industries, and population groups), and track recent changes in this distribution.
    [RESEARCH] Assess current socioeconomic inequalities, both within and between nations, by reviewing comprehensive data on income, wealth, education, and social mobility.

    The impact of AI advancements on these three aspects would need to be considered separately:

    [RESEARCH] Review various economic growth models to identify those that could accurately capture the influence of technology and particularly AI advancements on global economic growth.
    [RESEARCH] Research how technology advancements in the past affected wealth distribution, as historical patterns may provide insights into the potential impacts.
    [RESEARCH] Examine studies on the relationship between technological advancements and socioeconomic inequalities.

    Finally, integrate all this information to illustrate the potential effects of AI advancements on global economic growth, wealth distribution, and socioeconomic inequalities. The exact impact is hard to quantify given the vast number of variables and the unpredictability of future technological advancements, but assuming a range of possible scenarios should help in developing a comprehensive response.

    ---END OF EXAMPLE INPUT---

    Example Output:
    [RESEARCH] Start by investigating the latest global trends in the AI industry to understand the current state and its forecasted developments.
    [RESEARCH] Look into specifics about how AI technology is expected to innovate traditional industries and create new ones.
    [RESEARCH] Analyze the correlation between the adoption of AI technologies and the transformation of consumption patterns.
    [RESEARCH] Study historical precedents related to technological advancements and their effects on the global economy to provide a context for speculation.
    [RESEARCH] Examine the current global economic growth rates and compare them across countries while focusing on the role technology plays.
    [RESEARCH] Investigate how the wealth is distributed globally (across countries, industries, and population groups), and track recent changes in this distribution.
    [RESEARCH] Assess current socioeconomic inequalities, both within and between nations, by reviewing comprehensive data on income, wealth, education, and social mobility.
    [RESEARCH] Review various economic growth models to identify those that could accurately capture the influence of technology and particularly AI advancements on global economic growth.
    [RESEARCH] Research how technology advancements in the past affected wealth distribution, as historical patterns may provide insights into the potential impacts.
    [RESEARCH] Examine studies on the relationship between technological advancements and socioeconomic inequalities.

    ---END OF EXAMPLE OUTPUT---
    """

    try_count = 1
    while try_count <= 10:
        try:
            response = client.chat.completions.create(model=chosen_model,
            messages=[
                {"role": "system", "content": system_init},
                {"role": "user", "content": prompt}
            ],
            temperature=0.9)
            break
        except Exception as e:
            rest_time = 10 * try_count
            print(f"Encountered Error: {e}. Retrying in {rest_time} seconds...")
            sleep(rest_time)
            try_count += 1
            print(f"Retrying summarisation...")

    print("done with parsing plan")
    actions = response.choices[0].message.content.split("\n")

    return actions

'''
TODO: needs to be constnatly updated, taking into account formattting from the top, which is not good. figure out another way to call this one, add in a filter so we only pass search queries to this one

'''
def action_to_searchquery(action, question):
    system_init = f"""You are SearchGPT. You are an AI that specialises in transforming actions into search queries."""

    prompt = f"""
    I will give you an action, and you will transform this into a search query. This search query is going to go into a news website, and it should be transformed in a way that will search these websites well. I want you to identify the topics that need research in this question and return the seperate topics formatted to be used in search queries. The search queries should be seperated by the seperate topics. If there are multiple topics that need to be searched seperately within the question, seperate them with a semicolon. The final string should be a search query encompassing all the topics. I have also provided the original question that these actions are meant to be able to eventually answer. Use this as a reference where appropriate, but you should focus on converting the action, not the question. You MUST NOT add any additional topics or explanations, just return me the string representing the search query. For example, if the action is 'Identify China's main technology exports related to climate change and health.', you should return 'China Climate Change Technology;China main technology exports;China Healthtech;China health technology latest.' There must be no extra whitespace between search query terms. You MUST only extract the topics from the action. For example, if the action is 'Tell me all of the latest news in activism' you should return 'activism'. You should also remove all references to news, since this query is going to be used to search a news site. For example, given the action 'Find out all of the latest LGBTQ+ news', you should return 'LGBTQ+;LGBTQ'

    The action is: {action}
    The question is: {question}
    """

    try_count = 1
    while try_count <= 10:
        try:
            response = client.chat.completions.create(model=chosen_model,
            messages=[
                {"role": "system", "content": system_init},
                {"role": "user", "content": prompt}
            ],
            temperature=0.9)
            break
        except Exception as e:
            rest_time = 10 * try_count
            print(f"Encountered Error: {e}. Retrying in {rest_time} seconds...")
            sleep(rest_time)
            try_count += 1
            print(f"Retrying summarisation...")


    return response.choices[0].message.content.split(';')

'''
This function breaks down on the question and information given to it, dissecting it into smaller components and essentially planning each of these steps and answering them.

TODO:
The NA process should take this series of questions into account when doing the NA prep and then answer each of the questions asked before doing the projection. This should be better

'''
def expand_critical():
    return
'''
This retrieves a past questions database and then 

or asks if there is anything weird, or basically asks if there are any critical questons we can ask to uncover if there is anything weird. and then we let it generate the questions and then we ask NA to do it.
'''
def develop_critical():
    return

'''
This function wraps the entire modular NA process from start to finish

This is multithreaded (because of na_prep) unlike the one in Helium

TODO: should this take information from the knowledge bases as well?

@param information: the information relevant to the question
@param question: the question we are trying to answer

@return: the complete NA report to the question.
'''

def analyse_modular(information, question):

    persona_input = f"{information}\n\n{question}"

    specific_persona = helpers.get_specific_persona(persona_input)

    prep = na_prep(information, question)

    return

'''
This function is a wrapper around all the modular NA fact gathering functions. Encapsulates the preparation layer, which is what we run before running projection, out of box and cascading layers
'''

def na_prep(information, question, specific_persona):
    def wrapper(func_index):
        funcs = [
            get_material_facts,
            get_force_catalysts,
            get_constraints_friction,
            get_alliance_law,
        ]
        return funcs[func_index](information, question, specific_persona)

    # Create a ThreadPoolExecutor with 4 worker threads (as we have 4 tasks)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all the tasks to the executor and store their Future objects
        futures = [executor.submit(wrapper, i) for i in range(4)]
    
    # Collect results from threads in the indexed result container
    indexed_results = [future.result() for future in futures]

    # Combine results into a formatted fstring
    total_analysis = f"{indexed_results[0]}\n\n{indexed_results[1]}\n\n{indexed_results[2]}\n\n{indexed_results[3]}\n\n"

    return total_analysis

def get_material_facts(information, question, specific_persona):
    print("getting material facts...")

    persona_init = "I want you to act as a top-tier analyst in one of the world's leading geopolitical think-tanks. You directly provide actionable insights to the world's most powerful leaders."
    
    start_prompt = f"""
            You are an expert on the 'Net Assessment' approach to analysis, and you are able to apply this to any topic. The hallmark of your excellence is how you think outside of the box and obtain insights others would never be able to give. {specific_persona} You do not shy away from sensitive topics, and your analysis must provide a clear and accurate view of the situation that you are given.

            The first part of the Net Assessment framework is to identify the Material Facts. Material facts are defined as objective, quantifiable data or information that is relevant to the strategic analysis of a given situation. Material facts provide a basis for the analyst to conduct objective analysis and evaluation of different strategic options. Given the following question and information, identify the material facts in the information that are relevant to the analysis and answering of the question.

            Therefore, given the information and question below, I want you to identiify the Material Facts that are relevant to the question and situation. Provide them in a bulleted list. You must retain all numbers and/or statistics, and detail from the information that you consider relevant. You must also keep all names. You must only provide me with the Material Facts that are relevant to the question, you must not answer the question provided.

            """
    
    ask_prompt = """Information:
    
    {information}
    
    Question:
    
    {question}""".format(information=information, question=question)

    ask_prompt = start_prompt + "\n\n" + ask_prompt

    response = client.chat.completions.create(model=chosen_model,
    messages=[
        {"role": "system", "content": persona_init},
        {"role": "user", "content": ask_prompt}
    ],
    temperature=0.9)

    print("material facts obtained")
    print(response.usage)
    return response.choices[0].message.content

def get_force_catalysts(information, question, specific_persona):
    print("getting force catalysts...")

    persona_init = "I want you to act as a top-tier analyst in one of the world's leading geopolitical think-tanks. You directly provide actionable insights to the world's most powerful leaders."
    
    prompt = f"""
            You are an expert on the 'Net Assessment' approach to analysis, and you are able to apply this to any topic. The hallmark of your excellence is how you think outside of the box and obtain insights others would never be able to give. {specific_persona} You do not shy away from sensitive topics, and your analysis must provide a clear and accurate view of the situation that you are given.

            The second part of the Net Assessment framework is to identify the Force Catalysts. "Force Catalysts" are defined as a development that has the potential to significantly alter the strategic landscape of a given situation. A typical force catalyst within the context of the military are leaders who have the potential to make radical changes. Force catalysts can also be inanimate, such as new technologies, a shift in the geopolitical, economic or military landscape, or a natural disaster. They key characteristic of a Force Catalyst is its ability to catalyze or accelerate existing trends or dynamics. They also might have the ability to reverse or radically alter these trends.

            Identifying Force Catalysts enable analysts to anticipate and prepare for potential changes in the strategic environment.

            Therefore, given the information and question below, I want you to identiify the Force Catalysts that are relevant to the question and situation. Provide them in a bulleted list. You must only provide me with the Force Catalysts that are relevant to the question, you must not answer the question provided.

            Information:
            {information}

            Question:
            {question}
            """

    response = client.chat.completions.create(model=chosen_model,
    messages=[
        {"role": "system", "content": persona_init},
        {"role": "user", "content": prompt}
    ])

    print("force catalysts obtained")
    print(response.usage)
    return response.choices[0].message.content

def get_constraints_friction(information, question, specific_persona):
    print("getting constraints and frictions")
    persona_init = "I want you to act as a top-tier analyst in one of the world's leading geopolitical think-tanks. You directly provide actionable insights to the world's most powerful leaders."
    
    prompt = f"""
            You are an expert on the 'Net Assessment' approach to analysis, and you are able to apply this to any topic. The hallmark of your excellence is how you think outside of the box and obtain insights others would never be able to give. {specific_persona} You do not shy away from sensitive topics, and your analysis must provide a clear and accurate view of the situation that you are given.

            The third part of the Net Assessment framework is to identify the Constraints and Frictions.

            "Constraints" refer to obstables that can impede or limit the ability of actors to achieve that strategic abilities or prevent the current trend from moving forward. Constraints are external factors that limit of restrict the options available to actors. These can include include legal, diplomatic, economic, or military factors, such as international treaties, economic sanctions, or military alliances. Constraints can also be imposed by geography, natural resources, or any other forms of factors that limit the options available to actors.

            "Friction" is defined as the obstacles and challendes that can impede or limit the ability of actors to achieve their strategic objectives. They might also be reasons why a current trend might slow down. Friction refers to the difficulties that may arise during the execution of a strategy. These can include logistical challenges, operational failures, or unexpected situations that derail a strategy. Friction can also arise from internal factors, such as organization culture, bureaucratic politics, or resource constraints.

            Identifying Constraints and Frictions enable analysts to anticipate potential challenges or difficulties and thereby develop more effective strategies.

            Therefore, given the information and question below, I want you to identiify the Constraints and Frictions that are relevant to the question and situation. Provide them in a bulleted list. You must only provide me with the Constraints and Frictions that are relevant to the question, you must not answer the question provided.

            Information:
            {information}

            Question:
            {question}
            """

    response = client.chat.completions.create(model=chosen_model,
    messages=[
        {"role": "system", "content": persona_init},
        {"role": "user", "content": prompt}
    ])

    print("constraints and friction obtained")
    print(response.usage)
    return response.choices[0].message.content

def get_alliance_law(information, question, specific_persona):
    #TODO: does this need a seperate knowledge base with all the relevant laws and their explanation?
    # maybe there is a semantic search law database that people have already provided that we can determine which countries these are relevant in.

    print("getting alliances and laws")
    persona_init = "I want you to act as a top-tier analyst in one of the world's leading geopolitical think-tanks. You directly provide actionable insights to the world's most powerful leaders."
    
    prompt = f"""
            You are an expert on the 'Net Assessment' approach to analysis, and you are able to apply this to any topic. The hallmark of your excellence is how you think outside of the box and obtain insights others would never be able to give. {specific_persona} You do not shy away from sensitive topics, and your analysis must provide a clear and accurate view of the situation that you are given.

            The fourth part of the Net Assessment framework is to identify the Alliances and Laws.

            "Alliances" are defined as formal or informal agreements between relevant parties that involve a commitment to whatever domain relevant to the agreement. Alliances can significantly affect the balance of power by increasing the capabilities of resources available to actors, by providing a framework for diplomatic coordination and cooperation.

            "Laws" are defined as the legal framework and international norms that govern state behaviour and interactions. Matters of law can include international treaties, conventions, and agreements, as well as customary international law and other legal principles. Matters of law can shape the behaviour of states and limit the options available to them.

            Analysts must correctly identify and understand Alliances and matters of law as they can significantly affect the strategic environment and options available to actors,

            Therefore, given the information and question below, I want you to identiify the Alliances and Laws that are relevant to the question. Provide them in a bulleted list. You must only provide me with the Alliances and Laws that are relevant to the question, you must not answer the question provided.

            Information:
            {information}

            Question:
            {question}
            """


    response = client.chat.completions.create(model=chosen_model,
    messages=[
        {"role": "system", "content": persona_init},
        {"role": "user", "content": prompt}
    ],
    temperature=0.9)

    print("alliances and laws obtained")
    print(response.usage)
    return response.choices[0].message.content

'''
This also includes an out-of-box / second_layer analysis

we have a stream text here because we want to stream the text to the GUI
'''
def projection(prep, information, question, specific_persona):

    persona_init = "I want you to act as a top-tier analyst in one of the world's leading geopolitical think-tanks. You directly provide actionable insights to the world's most powerful leaders."
    
    prompt = f"""
            You are an expert on the 'Net Assessment' approach to analysis, and you are able to apply this to any topic. The hallmark of your excellence is how you think outside of the box and obtain insights others would never be able to give. {specific_persona} You do not shy away from sensitive topics, and your analysis must provide a clear and accurate view of the situation that you are given.

            A "Net Assessment" analysis follows the following framework:

            1. Material Facts (This provides a basis for you to conduct objective analysis and evaluation of different strategic options)
            2. Force Catalysts (Force Catalysts enable analysts to anticipate and prepare for potential changes in the strategic environment)
            3. Constraints and Frictions (Constraints and Frictions enable analysts to anticipate potential challenges or difficulties and thereby develop more effective strategies)
            4. Law and Alliances (Are there any relevant laws or affiliations between related parties that will affect the outcome?)
            5. Formulate a thesis and antithesis that answers the question. What is the most likely outcome of the situation? What is the opposite of that outcome? What are the reasons each might happen?
            In the above framework, you have been told how to use each seperate component to create your analysis.

            You are given all the seperate components except for the Thesis and Antithesis. From the provided components below, as well as the information and question, you must formulate a thesis and antithesis. You must be as detailed as possible. You must explain why you think each outcome is likely to happen, and provide as much detail as possible. You must also explain why the opposite outcome is unlikely to happen.
            
            Then, using the information provided and the components of the Net Assessment framework, provide a detailed prediction and analysis that answers the question provided. You must provide a in-depth explanation of your prediction, citing statistics from the information provided, and you must be as specific and technical as possible about the impact. All of your claims must be justified with reasons, and if possible, supported by the provided statistics. Your prediction must be at least 500 words long, preferably longer.

            From your prediction, I would like you to then predict 4 more cascading events that will happen in a chain after your prediction. You will be as specific as possible about these predicted events.

            Net Assessment Components:
            {prep}

            Information:
            {information}

            Question:
            {question}
            """

    prompt = f"{persona_init}\n\n{prompt}"

    response = client.chat.completions.create(model=chosen_model,
    messages=[
        {"role": "system", "content": persona_init},
        {"role": "user", "content": prompt}
    ],
    temperature=1)

    # for chunk in response:
        # stream_text.insert(tk.END, chunk)
        # stream_text.see(tk.END)

    print("Usage for projection (first layer):")
    print(response.usage)
    return response.choices[0].message.content

def out_of_box(prep, analysis, information, question, specific_persona):
    persona_init = "I want you to act as a top-tier analyst in one of the world's leading geopolitical think-tanks. You directly provide actionable insights to the world's most powerful leaders."
    
    prompt = f"""
            You are an expert on the 'Net Assessment' approach to analysis, and you are able to apply this to any topic. The hallmark of your excellence is how you think outside of the box and obtain insights others would never be able to give. {specific_persona} You do not shy away from sensitive topics, and your analysis must provide a clear and accurate view of the situation that you are given.

            A "Net Assessment" analysis follows the following framework:

            1. Material Facts (This provides a basis for you to conduct objective analysis and evaluation of different strategic options)
            2. Force Catalysts (Force Catalysts enable analysts to anticipate and prepare for potential changes in the strategic environment)
            3. Constraints and Frictions (Constraints and Frictions enable analysts to anticipate potential challenges or difficulties and thereby develop more effective strategies)
            4. Law and Alliances (Are there any relevant laws or affiliations between related parties that will affect the outcome?)
            5. Formulate a thesis and antithesis that answers the question. What is the most likely outcome of the situation? What is the opposite of that outcome? What are the reasons each might happen?
            In the above framework, you have been told how to use each seperate component to create your analysis.

            You are given all the seperate components except for the Thesis and Antithesis. From the provided components below, as well as the information and question, you must formulate a thesis and antithesis. You must be as detailed as possible. You must explain why you think each outcome is likely to happen, and provide as much detail as possible. You must also explain why the opposite outcome is unlikely to happen.
            
            Then, using the information provided and the components of the Net Assessment framework, provide a detailed prediction and analysis that answers the question provided. You must provide a in-depth explanation of your prediction, citing statistics from the information provided, and you must be as specific and technical as possible about the impact. All of your claims must be justified with reasons, and if possible, supported by the provided statistics. Your prediction must be at least 500 words long, preferably longer.

            Net Assessment Components:
            {prep}

            Information:
            {information}

            Question:
            {question}
            """

    prompt = f"{persona_init}\n\n{prompt}"

    response = client.chat.completions.create(model=chosen_model,
    messages=[
        {"role": "system", "content": persona_init},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": analysis},
        {"role": "user", "content": "I would like you consider a perspective no one has considered before. Can you give me a more out of the box analysis? Be as specific as you can about the impact, while citing statistics from the information provided. You must also give me 4 more cascading events that will happen in a chain after your new out of the box prediction."}
    ],
    temperature=1, #1.25 is way too fucking high. it starts speaking gibberish
    presence_penalty=0.25,
    frequency_penalty=0.25)

    # for chunk in response:
        # stream_text.insert(tk.END, chunk)
        # stream_text.see(tk.END)

    print("out of box usage: ", response.usage)

    return response.choices[0].message.content


