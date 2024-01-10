import openai

def LLM_evaluator(node, goal, model):
    prompt = f"""
        You are an AI judge assigned to evaluate the likelihood of a user finding a specific goal within a described scene. All relevant details about the goal and the scene will be provided in text format. Based on the information, you are to assign a Likert score ranging from 1 to 5. Here's what each score represents:

        1: Highly unlikely the user will find the goal.
        2: Unusual scenario, but there's a chance.
        3: Equal probability of finding or not finding the goal.
        4: Likely the user will find the goal.
        5: Very likely the user will find the goal.

        If the scene is largely object or walls, means you are about to hit something, give a score of 1 this case.

        Your response should only be the score (a number between 1 and 5) without any additional commentary. For instance, if it's very likely, simply reply with "5".

        User's goal: {goal}
        Described scene:
        """ + str(node)
    # print("PROMPT::::", prompt)
    message=[{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages = message,
        temperature=0.8,
        max_tokens=500,
        frequency_penalty=0.0
    )
    # print(f"Node: {node}")
    try:
        score = int(response['choices'][0]['message']['content'].strip())
    except ValueError:
        # Handle unexpected output
        score = 3  # Default to a neutral score or handle differently as needed    
    print("Score:", score)
    return score

def LLM_world_model(node, model):
    prompt = f"""
    You are an AI tasked with extrapolating from a given scene description. Based on the details provided in this scene, envision what this setting could evolve into if one were to continue forward. Your response should be a detailed scene description, focusing on potential elements that may logically appear ahead based on the current context.

    You should provide a concise description of the given environment that don't exceed 100 words. Emphasize physical structures and natural elements, ensuring specific details about their conditions and characteristics are included. Refrain from mentioning people or activities

    An example of the answer: Moving forward from the curb pavement, the walkway broadens into a cobblestone plaza. Ahead, dense trees form a shaded canopy, under which modern LED streetlights stand at regular intervals. The path leads to a large stone fountain surrounded by a manicured lawn and symmetrical plant beds filled with seasonal flowers. The cobblestone trail continues, branching into multiple paths lined with metal bicycle racks.
    
    Current scene observation: {node}
    """
    message=[{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages = message,
        temperature=0.5,
        max_tokens=3000,
        frequency_penalty=0.0
    )
    # print(f"Current scene observation: {node}")
    extrapolated_scene = response.choices[0].message['content'].strip()
    # print("Extrapolated scene:", extrapolated_scene)

    return extrapolated_scene

def LLM_checker(node, goal, model):
    prompt = f"""
    You are tasked to check if the given scene descriptions have the goal object to find. The secene descriptions: {node} The Goal: {goal}. If you find the goal in the scene, please return yse, and if you didn't find goal in the scene, return no. The answer has to only be yes or no and nothing else. '
    """    
    message=[{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages = message,
        temperature=0.9,
        max_tokens=20,
        frequency_penalty=0.0
    )

    check = response.choices[0].message['content'].strip()

    return check
