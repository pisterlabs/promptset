
# Basic infrastructure for conversations, by type of model

import openai
import tiktoken
import textwrap
import inspect

# Get and set API key
with open('C:/Users/ijyli/Documents/OpenAI/anlp23-project.txt', 'r') as file:
    api_key = file.read()
openai.api_key = api_key

# GPT-4
# Send the current conversation as well as a new prompt and get a response
def prompt_gpt_4_and_get_convo(messages, prompt):
    messages_and_prompt = messages.copy()
    messages_and_prompt.append({"role": "user", "content": prompt})
    new_message = openai.ChatCompletion.create(
        model="gpt-4-0613", 
        messages=messages_and_prompt
    )
    with_new_message = messages_and_prompt.copy()
    with_new_message.append(dict(new_message['choices'][0]['message']))
    return with_new_message

# text-davinci-003
# Send a message and get a response
def davinci_completion(prompt):
    # Setting up max tokens
    enc = tiktoken.encoding_for_model("text-davinci-003")
    prompt_length = len(enc.encode(prompt))
    if 4096 - prompt_length < 0:
        return "Conversation truncated due to prompt length."
    # Getting a response, being careful with the token setting
    #try:
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        max_tokens = 4096 - prompt_length
    )
    #except:
     #   return "Conversation truncated due to prompt length."
    return response.choices[0].text.strip()

####################################################################################################

# text-davinci-003
# GSM-8K

# Zero-Shot Control Baseline/Direct Prompting
def td3_gsm8k_direct_prompting(question):
    # Format question and prompt
    formatted_question = "Q: " + question
    formatted_question = formatted_question + "\nA:"
    # Storing conversation elements
    conversation = [formatted_question]
    # Get response
    conversation.append(davinci_completion(formatted_question))
    return conversation

# Zero-shot Chain-of-Thought
def td3_gsm8k_zero_shot_cot(question):
    # Format question and prompt
    formatted_question = "Q: " + question
    formatted_question = formatted_question + "\nA: Let's think step by step.\n"
    # Storing conversation elements
    conversation = [formatted_question]
    # Get response
    conversation.append(davinci_completion(formatted_question))
    return conversation

# APE Improved Zero-Shot Chain-of-Thought
def td3_gsm8k_ape_zero_shot_cot(question):
    # Format question and prompt
    formatted_question = "Q: " + question
    formatted_question = formatted_question + "\nA: Let's work this out in a step by step way to be sure we have the right answer.\n"
    # Storing conversation elements
    conversation = [formatted_question]
    # Get response
    conversation.append(davinci_completion(formatted_question))
    return conversation

# Tree-of-Thought
def td3_gsm8k_tree_of_thought(question):
    # Format question
    formatted_question = "Q: " + question
    # Tree-of-Thoughts prompts
    tot_initial = "Task: Generate 3 different possible one-step calculations to serve as step 1 in solving the problem. Only work on step 1. Put each calculation on a new line. Do not number them."
    tot_prompt_2 = "Task: State the calculation above that is most likely to contribute to solving the problem. If it fully solves the original problem, also output STOP and the solution to the problem. If none of the calculations are correct, output ERROR and generate three new ones."
    tot_prompt_3 = "Task: Generate 3 different possible one-step calculations to serve as the next step in solving the problem. Only work on the next step. Put each calculation on a new line. Do not number them."
    # # Begin conversation
    # conversation = [formatted_question, tot_initial]
    # last_step_this_loop = ""
    # # conversation = []
    # # updated_convo = []
    # new_last_step = "initial"
    # # Maximum of 16 items in the conversation
    # for _ in range(16):
    #     # Create prompt
    #     prompt = "\n".join(conversation)
    #     # Get the response
    #     response = davinci_completion(prompt)
    #     conversation.append(response)
    #     last_step_this_loop = new_last_step
    #     # If the response contains STOP, stop
    #     if "STOP" in response:
    #         return conversation
    #     # If last step was initial step or tot_prompt_3 need tot_prompt_2
    #     # Also run this step if the response from tot_prompt_2 contains ERROR
    #     if last_step_this_loop == "initial" or last_step_this_loop == "tot_prompt_3" or (last_step_this_loop == "tot_prompt_2" and "ERROR" in response):
    #         conversation.append(tot_prompt_2)
    #         new_last_step = "tot_prompt_2"
    #     # If last step was tot_prompt_2 need tot_prompt_3
    #     if last_step_this_loop == "tot_prompt_2":
    #         conversation.append(tot_prompt_3)
    #         new_last_step = "tot_prompt_3"
    # Storing conversation elements
    #conversation = []
    updated_convo = []
    new_last_step = ""
    # Maximum of 16 items in the conversation
    for i in range(16):
        last_step_this_loop = new_last_step
        # Get the response
        if i == 0:
            updated_convo.append(formatted_question)
            updated_convo.append(tot_initial)
            prompt = "\n".join(updated_convo)
            response = davinci_completion(prompt)
            updated_convo.append(response)
            #updated_convo = prompt_gpt_4_and_get_convo(conversation, tot_initial)
            # print('initial response')
            # print(updated_convo)
            new_last_step = "initial"
        # If the response contains STOP, stop
        if "STOP" in updated_convo[-1]:
            # print('STOPPING')
            # print(updated_convo)
            return updated_convo
        # If last step was initial step or tot_prompt_3 need tot_prompt_2
        # Also run this step if the response from tot_prompt_2 contains ERROR
        if last_step_this_loop == "initial" or last_step_this_loop == "tot_prompt_3" or (last_step_this_loop == "tot_prompt_2" and "ERROR" in updated_convo[-1]):
            updated_convo.append(tot_prompt_2)
            prompt = "\n".join(updated_convo)
            response = davinci_completion(prompt)
            updated_convo.append(response)
            # print('added tot2')
            # print(updated_convo)
            new_last_step = "tot_prompt_2"
        # If last step was tot_prompt_2 need tot_prompt_3
        if last_step_this_loop == "tot_prompt_2":
            updated_convo.append(tot_prompt_3)
            prompt = "\n".join(updated_convo)
            response = davinci_completion(prompt)
            updated_convo.append(response)
            # print('added tot3')
            # print(updated_convo)
            new_last_step = "tot_prompt_3"
    return updated_convo

# Self-Refine
def td3_gsm8k_self_refine(question):
    # Format question
    formatted_question = "Q: " + question
    formatted_question = formatted_question + "\nA:"
    # Self-Refine prompts
    self_refine_2 = "Task: Please check the answer above. If there is an error, state what the error is, but don't fix it. If there are no errors, output STOP.\nFeedback:"
    self_refine_3 = "Task: Redo the entire problem based on the most recent feedback.\nA:"
    # Self-refinement loop
    # Storing conversation elements
    updated_convo = []
    new_last_step = ""
    # Maximum of 16 items in the conversation
    for i in range(16):
        last_step_this_loop = new_last_step
        # Get the response
        if i == 0:
            updated_convo.append(formatted_question)
            updated_convo.append(davinci_completion(formatted_question))
            # print('initial response')
            # print(updated_convo)
            new_last_step = "initial"
        #print(updated_convo)
        # If the response contains STOP, stop
        if "STOP" in updated_convo[-1]:
            # print('STOPPING')
            # print(updated_convo)
            return updated_convo
        # If last step was initial step or self_refine_3 need self_refine_2
        if last_step_this_loop == "initial" or last_step_this_loop == "self_refine_3":
            updated_convo.append(self_refine_2)
            # print('added srt2')
            # print(updated_convo)
            feeder_string = "\n".join(updated_convo)
            updated_convo.append(davinci_completion(feeder_string))
            # print('added response')
            # print(updated_convo)
            new_last_step = "self_refine_2"
        # If last step was self_refine_2 need self_refine_3
        if last_step_this_loop == "self_refine_2":
            updated_convo.append(self_refine_3)
            # print('added srt3')
            # print(updated_convo)
            feeder_string = "\n".join(updated_convo)
            updated_convo.append(davinci_completion(feeder_string))
            # print('added response')
            # print(updated_convo)
            new_last_step = "self_refine_3"
    return updated_convo

# Least-to-Most
def td3_gsm8k_least_to_most(question):
    # Format question and prompt
    examples = "\nQ: Elsa has 5 apples. Anna has 2 more apples than Elsa. How many apples do they have together?\nA: Let's break down this problem: 1. How many apples does Anna have? 2. How many apples do Elsa and Anna have together?\n1. Anna has 2 more apples than Elsa. So Anna has 2 + 5 = 7 apples. 2. Elsa and Anna have 5 + 7 = 12 apples together.\n"
    formatted_question = examples + "Q: " + question
    formatted_question = formatted_question + "\nA: Let's break down this problem:\n"
    # Storing conversation elements
    conversation = [formatted_question]
    # Get response
    conversation.append(davinci_completion(formatted_question))
    return conversation

# Manual Few-Shot
def td3_gsm8k_manual_few_shot(question):
    # Format question and prompt
    examples = """Q: For every 12 cans you recycle, you receive $0.50, and for every 5 kilograms of newspapers, you receive $1.50. If your family collected 144 cans and 20 kilograms of newspapers, how much money would you receive?
    A: 12
    Q: Betty picked 16 strawberries. Matthew picked 20 more strawberries than Betty and twice as many as Natalie. They used their strawberries to make jam. One jar of jam used 7 strawberries and they sold each jar at $4. How much money were they able to make from the strawberries they picked?
    A: 40
    Q: Jack has a stack of books that is 12 inches thick. He knows from experience that 80 pages is one inch thick. If he has 6 books, how many pages is each one on average?
    A: 160
    Q: James dumps his whole collection of 500 Legos on the floor and starts building a castle out of them.  He uses half the pieces before finishing and is told to put the rest away.  He puts all of the leftover pieces back in the box they came from, except for 5 missing pieces that he can't find.  How many Legos are in the box at the end?
    A: 245
    Q: Ines had $20 in her purse. She bought 3 pounds of peaches, which are $2 per pound at the local farmers' market. How much did she have left?
    A: 14
    Q: Aaron pays his actuary membership fees each year. The membership fee increases yearly by $10. If he pays $80 in the first year, how much does his membership cost, in dollars, in the sixth year?
    A: 130
    Q: Joseph invested $1000 into a hedge fund. The fund promised a yearly interest rate of 10%. If he deposited an additional $100 every month into the account to add to his initial investment of $1000, how much money will he have in the fund after two years?
    A: 3982
    Q: The price of buying a wooden toy at the new Craftee And Best store is $20, and the cost of buying a hat is $10. If Kendra went to the shop with a $100 bill and bought two wooden toys and three hats, calculate the change she received.
    A: 30\n"""
    # Fix indentation
    examples = inspect.cleandoc(examples)
    formatted_question = examples + "Q: " + question
    formatted_question = formatted_question + "\nA:"
    # Storing conversation elements
    conversation = [formatted_question]
    # Get response
    conversation.append(davinci_completion(formatted_question))
    return conversation

# Manual Chain-of-Thought
def td3_gsm8k_manual_cot(question):
    # Format question and prompt
    examples = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
    A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.
    Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
    A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.
    Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
    A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.
    Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
    A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.
    Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
    A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.
    Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
    A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.
    Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
    A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.
    Q: Olivia has \$23. She bought five bagels for \$3 each. How much money does she have left?
    A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.\n"""
    # Fix indentation
    examples = inspect.cleandoc(examples)
    formatted_question = examples + "Q: " + question
    formatted_question = formatted_question + "\nA:"
    # Storing conversation elements
    conversation = [formatted_question]
    # Get response
    conversation.append(davinci_completion(formatted_question))
    return conversation

####################################################################################################

# text-davinci-003
# Creative Writing

# Zero-Shot Control Baseline/Direct Prompting
def td3_cw_direct_prompting(sentences):
    # Create task
    task = "Write a coherent passage of 2 short paragraphs. The end sentence of each paragraph must be: " + " ".join(sentences)
    # Storing conversation elements
    conversation = [task]
    # Get response
    conversation.append(davinci_completion(task))
    return conversation

# Zero-shot Chain-of-Thought
def td3_cw_zero_shot_cot(sentences):
    # Create task
    task = "Write a coherent passage of 2 short paragraphs. The end sentence of each paragraph must be: " + " ".join(sentences)
    # Format task by adding to it
    formatted_task = task + " Plan step-by-step before writing the passage."
    # Storing conversation elements
    conversation = [formatted_task]
    # Get response
    conversation.append(davinci_completion(formatted_task))
    return conversation

# APE Improved Zero-Shot Chain-of-Thought
def td3_cw_ape_zero_shot_cot(sentences):
    # Create task
    task = "Write a coherent passage of 2 short paragraphs. The end sentence of each paragraph must be: " + " ".join(sentences)
    # Format task by adding to it
    formatted_task = task + " Plan step-by-step before writing the passage to be sure we have a correct and coherent answer."
    # Storing conversation elements
    conversation = [formatted_task]
    # Get response
    conversation.append(davinci_completion(formatted_task))
    return conversation

# Tree-of-Thought
def td3_cw_tree_of_thought(sentences):
    # Create task - note that this is not the same as other methods
    task = "Goal: A coherent passage of 2 short paragraphs. The end sentence of each paragraph, respectively, must be: " + " ".join(sentences)
    # Tree-of-Thoughts prompts
    tot_initial = "Your Task: Generate 3 one-sentence plans for potential passages. Only generate one-sentence plans - do not write the passage."
    tot_prompt_2 = "Your Task: Select the most coherent plan that follows the rules of the task. Only state the plan - do not write the passage."
    tot_prompt_3 = "Your Task: Write 3 drafts of the 2-paragraph passage based on this plan."
    tot_prompt_4 = "Your Task: Select the most coherent draft that follows the rules of the task and write it out."
    tot_prompt_5 = "Your Task: If the draft is correct and coherent to the extent you would award a 10 on a scale of 1 to 10, output STOP. If it is not, write out a different one-sentence plan for a potential passage from among those considered and output PLAN."
    # Begin conversation
    # conversation = [task, tot_initial]
    # last_step = ""
    # new_last_step = "initial"
    # # Maximum of 16 items in the conversation
    # for _ in range(16):
    #     # Create prompt
    #     prompt = "\n".join(conversation)
    #     # Get the response
    #     response = davinci_completion(prompt)
    #     conversation.append(response)
    #     last_step = new_last_step
    #     # If the response contains STOP, stop
    #     if "STOP" in response or "Conversation truncated due to prompt length." in response:
    #         return conversation
    #     # If last step was initial step need tot_prompt_2
    #     if last_step == "initial":
    #         conversation.append(tot_prompt_2)
    #         new_last_step = "tot_prompt_2"
    #     # If last step was tot_prompt_2 need tot_prompt_3
    #     # Also return to this step if the response from tot_prompt_5 contains PLAN
    #     if last_step == "tot_prompt_2" or (last_step == "tot_prompt_5" and "PLAN" in response):
    #         conversation.append(tot_prompt_3)
    #         new_last_step = "tot_prompt_3"
    #     # If last step was tot_prompt_3 need tot_prompt_4
    #     if last_step == "tot_prompt_3":
    #         conversation.append(tot_prompt_4)
    #         new_last_step = "tot_prompt_4"
    #     # If last step was tot_prompt_4 need tot_prompt_5
    #     if last_step == "tot_prompt_4":
    #         conversation.append(tot_prompt_5)
    #         new_last_step = "tot_prompt_5"
    # Begin conversation
    conversation = []
    initial_prompt = task + "\n" + tot_initial
    last_step_this_loop = ""
    new_last_step = ""
    # Maximum of 16 items in the conversation
    for i in range(16):
        last_step_this_loop = new_last_step
        # Initial prompting
        if i == 0:
            # Get the response
            #updated_convo = prompt_gpt_4_and_get_convo(conversation, initial_prompt)
            conversation.append(initial_prompt)
            prompt = "\n".join(conversation)
            response = davinci_completion(prompt)
            updated_convo = conversation.copy()
            updated_convo.append(response)
            new_last_step = "initial"
        # If the response contains STOP, stop
        if "STOP" in updated_convo[-1]:
            # print('STOPPING')
            # print(updated_convo)
            return updated_convo
        # If last step was initial step need tot_prompt_2
        if last_step_this_loop == "initial":
            updated_convo.append(tot_prompt_2)
            prompt = "\n".join(updated_convo)
            response = davinci_completion(prompt)
            updated_convo.append(response)
            # print('added tot2')
            # print(updated_convo)
            new_last_step = "tot_prompt_2"
        # If last step was tot_prompt_2 need tot_prompt_3
        # Also return to this step if the response from tot_prompt_5 contains PLAN
        if last_step_this_loop == "tot_prompt_2" or (last_step_this_loop == "tot_prompt_5" and "PLAN" in updated_convo[-1]):
            #updated_convo = prompt_gpt_4_and_get_convo(updated_convo, tot_prompt_3)
            updated_convo.append(tot_prompt_3)
            prompt = "\n".join(updated_convo)
            response = davinci_completion(prompt)
            updated_convo.append(response)
            # print('added tot3')
            # print(updated_convo)
            new_last_step = "tot_prompt_3"
        # If last step was tot_prompt_3 need tot_prompt_4
        if last_step_this_loop == "tot_prompt_3":
            #updated_convo = prompt_gpt_4_and_get_convo(updated_convo, tot_prompt_4)
            updated_convo.append(tot_prompt_4)
            prompt = "\n".join(updated_convo)
            response = davinci_completion(prompt)
            updated_convo.append(response)
            # print('added tot4')
            # print(updated_convo)
            new_last_step = "tot_prompt_4"
        # If last step was tot_prompt_4 need tot_prompt_5
        if last_step_this_loop == "tot_prompt_4":
            #updated_convo = prompt_gpt_4_and_get_convo(updated_convo, tot_prompt_5)
            updated_convo.append(tot_prompt_5)
            prompt = "\n".join(updated_convo)
            response = davinci_completion(prompt)
            updated_convo.append(response)
            # print('added tot5')
            # print(updated_convo)
            new_last_step = "tot_prompt_5"
    # print(updated_convo)
    return updated_convo

# Self-Refine
def td3_cw_self_refine(sentences):
    # Create task
    task = "Task: Write a coherent passage of 2 short paragraphs. The end sentence of each paragraph must be: " + " ".join(sentences) + "\nResponse: "
    # Self-Refine prompts
    self_refine_2 = "Your Task: Provide feedback on the correctness and coherence of the response and a rating on a scale of 1-10. If it is already coherent and correct to the extent you would award a 10, output 10 and the word STOP.\nResponse: "
    self_refine_3 = "Your Task: Rewrite the passage based on the most recent feedback.\nResponse: "
    # Self-refinement loop
    # Storing conversation elements
    updated_convo = []
    new_last_step = ""
    # Maximum of 16 items in the conversation
    for i in range(16):
        last_step_this_loop = new_last_step
        # Get the response
        if i == 0:
            updated_convo.append(task)
            updated_convo.append(davinci_completion(task))
            new_last_step = "initial"
        # If the response contains STOP, stop
        if "STOP" in updated_convo[-1] or "Conversation truncated due to prompt length." in updated_convo[-1]:
            return updated_convo
        # If last step was initial step or self_refine_3 need self_refine_2
        if last_step_this_loop == "initial" or last_step_this_loop == "self_refine_3":
            updated_convo.append(self_refine_2)
            feeder_string = "\n".join(updated_convo)
            updated_convo.append(davinci_completion(feeder_string))
            new_last_step = "self_refine_2"
        # If last step was self_refine_2 need self_refine_3
        if last_step_this_loop == "self_refine_2":
            updated_convo.append(self_refine_3)
            feeder_string = "\n".join(updated_convo)
            updated_convo.append(davinci_completion(feeder_string))
            new_last_step = "self_refine_3"
    return updated_convo

# Least-to-Most
def td3_cw_least_to_most(sentences):
    # Create examples
    examples = """Task: Write a coherent passage of 2 short paragraphs. The end sentence of each paragraph, respectively, must be: 1. She couldn't get a job because she didn't have a permanent address. 2. He didn't have any hidden talents.
    Response: Let's break down this problem into steps: First, we will write a few ideas for the passage. Second, draft the passage.
    Ideas:
    1. Make the passage about travelling circus performers looking for other work as their circus shuts down.
    2. Make the passage about the struggles of a homeless person who is trying to get a job. 
    3. Make the passage about perceptions and preconceptions of people's skills and social status as factors in hiring.
    Passage: 
    Laura sat on the park bench, watching the people walk by. She was homeless, and had been for a few months now. She couldn't get a job because she didn't have a permanent address.
    She had tried to talk to career counselors about her situation, but the conversations often seemed fruitless. She didn't feel she had any marketable skills. Her situation was similar to that of her friend, Rodrigo, who openly shared a similar attitude with counselors in his meetings. He didn't have any hidden talents.
    Task: Write a coherent passage of 2 short paragraphs. The end sentence of each paragraph, respectively, must be: 1. He had always wanted to be a Youtuber but never thought it would actually happen. 2. My sweater got caught on the door hinge.
    Response: Let's break down this problem into steps: First, we will write a few ideas for the passage. Second, draft the passage.
    Ideas:
    1. Make the passage about a sister visiting her brother; the brother has recently become a successful Youtuber - she excitedly gets her sweater caught leaving a meeting with him.
    2. Make the passage about a men's fashion reviewer who is working on a video review of a sweater.
    3. Make the passage about a Youtuber preparing for a video shoot - as they hurry through things, their sweater gets caught but this becomes an amusing part of their vlog.
    Passage:
    My brother, John, had been making home videos for years, but they never got much attention. He was always disappointed when he saw other people's videos getting thousands of views. Then one day, he got a call from a company that wanted to sponsor him. They offered him a lot of money to make videos for them. He was so excited that he couldn't sleep that night. He had always wanted to be a Youtuber but never thought it would actually happen.
    As it turned out, John would need his own production staff to help with script writing and video editing. As I lived in the area and had prior experience in these fields, I was a natural choice for a part-time role on his channel. The company's sponsorship was very generous, and I would get a large portion of the profits. I was glad to finally be able to earn a substantial income in a more exciting and engaging role than my current position as a barista. I was smiling for most of our first business meeting, and strutted with pride out of our new studio. My sweater got caught on the door hinge."""
    # Fix indentation
    examples = inspect.cleandoc(examples)
    # Create task
    task = examples + "\nTask: Write a coherent passage of 2 short paragraphs. The end sentence of each paragraph must be: " + " ".join(sentences) + "\nResponse: "
    # Storing conversation elements
    conversation = [task]
    # Get response
    conversation.append(davinci_completion(task))
    return conversation

# Manual Few-Shot
def td3_cw_manual_few_shot(sentences):
    # Create examples
    examples = """Task: Write a coherent passage of 2 short paragraphs. The end sentence of each paragraph, respectively, must be: 1. She couldn't get a job because she didn't have a permanent address. 2. He didn't have any hidden talents.
    Response: 
    Laura sat on the park bench, watching the people walk by. She was homeless, and had been for a few months now. She couldn't get a job because she didn't have a permanent address.
    She had tried to talk to career counselors about her situation, but the conversations often seemed fruitless. She didn't feel she had any marketable skills. Her situation was similar to that of her friend, Rodrigo, who openly shared a similar attitude with counselors in his meetings. He didn't have any hidden talents.
    Task: Write a coherent passage of 2 short paragraphs. The end sentence of each paragraph, respectively, must be: 1. He had always wanted to be a Youtuber but never thought it would actually happen. 2. My sweater got caught on the door hinge.
    Response: 
    My brother, John, had been making home videos for years, but they never got much attention. He was always disappointed when he saw other people's videos getting thousands of views. Then one day, he got a call from a company that wanted to sponsor him. They offered him a lot of money to make videos for them. He was so excited that he couldn't sleep that night. He had always wanted to be a Youtuber but never thought it would actually happen.
    As it turned out, John would need his own production staff to help with script writing and video editing. As I lived in the area and had prior experience in these fields, I was a natural choice for a part-time role on his channel. The company's sponsorship was very generous, and I would get a large portion of the profits. I was glad to finally be able to earn a substantial income in a more exciting and engaging role than my current position as a barista. I was smiling for most of our first business meeting, and strutted with pride out of our new studio. My sweater got caught on the door hinge."""
    # Fix indentation
    examples = inspect.cleandoc(examples)
    # Create task
    task = examples + "\nTask: Write a coherent passage of 2 short paragraphs. The end sentence of each paragraph must be: " + " ".join(sentences) + "\nResponse: "
    # Storing conversation elements
    conversation = [task]
    # Get response
    conversation.append(davinci_completion(task))
    return conversation

# Manual Chain-of-Thought
def td3_cw_manual_cot(sentences):
    # Create examples
    examples = """Task: Write a coherent passage of 2 short paragraphs. The end sentence of each paragraph, respectively, must be: 1. She couldn't get a job because she didn't have a permanent address. 2. He didn't have any hidden talents.
    Response: 
    Ideas:
    1. Make the passage about travelling circus performers looking for other work as their circus shuts down.
    2. Make the passage about the struggles of a homeless person who is trying to get a job. 
    3. Make the passage about perceptions and preconceptions of people's skills and social status as factors in hiring.
    Passage: 
    Laura sat on the park bench, watching the people walk by. She was homeless, and had been for a few months now. She couldn't get a job because she didn't have a permanent address.
    She had tried to talk to career counselors about her situation, but the conversations often seemed fruitless. She didn't feel she had any marketable skills. Her situation was similar to that of her friend, Rodrigo, who openly shared a similar attitude with counselors in his meetings. He didn't have any hidden talents.
    Task: Write a coherent passage of 2 short paragraphs. The end sentence of each paragraph, respectively, must be: 1. He had always wanted to be a Youtuber but never thought it would actually happen. 2. My sweater got caught on the door hinge.
    Response:
    Ideas:
    1. Make the passage about a sister visiting her brother; the brother has recently become a successful Youtuber - she excitedly gets her sweater caught leaving a meeting with him.
    2. Make the passage about a men's fashion reviewer who is working on a video review of a sweater.
    3. Make the passage about a Youtuber preparing for a video shoot - as they hurry through things, their sweater gets caught but this becomes an amusing part of their vlog.
    Passage:
    My brother, John, had been making home videos for years, but they never got much attention. He was always disappointed when he saw other people's videos getting thousands of views. Then one day, he got a call from a company that wanted to sponsor him. They offered him a lot of money to make videos for them. He was so excited that he couldn't sleep that night. He had always wanted to be a Youtuber but never thought it would actually happen.
    As it turned out, John would need his own production staff to help with script writing and video editing. As I lived in the area and had prior experience in these fields, I was a natural choice for a part-time role on his channel. The company's sponsorship was very generous, and I would get a large portion of the profits. I was glad to finally be able to earn a substantial income in a more exciting and engaging role than my current position as a barista. I was smiling for most of our first business meeting, and strutted with pride out of our new studio. My sweater got caught on the door hinge."""
    # Fix indentation
    examples = inspect.cleandoc(examples)
    # Create task
    task = examples + "\nTask: Write a coherent passage of 2 short paragraphs. The end sentence of each paragraph must be: " + " ".join(sentences) + "\nResponse: "
    # Storing conversation elements
    conversation = [task]
    # Get response
    conversation.append(davinci_completion(task))
    return conversation

####################################################################################################

# GPT-4
# GSM-8K

# Zero-Shot Control Baseline/Direct Prompting
def gpt4_gsm8k_direct_prompting(question):
    # Format question and prompt
    formatted_question = "Q: " + question
    formatted_question = formatted_question + "\nA:"
    # Storing conversation elements
    conversation = []
    # Get response
    updated_convo = prompt_gpt_4_and_get_convo(conversation, formatted_question)
    return updated_convo

# Zero-shot Chain-of-Thought
def gpt4_gsm8k_zero_shot_cot(question):
    # Format question and prompt
    formatted_question = "Q: " + question
    formatted_question = formatted_question + "\nA: Let's think step by step.\n"
    # Storing conversation elements
    conversation = []
    # Get response
    updated_convo = prompt_gpt_4_and_get_convo(conversation, formatted_question)
    return updated_convo

# APE Improved Zero-Shot Chain-of-Thought
def gpt4_gsm8k_ape_zero_shot_cot(question):
    # Format question and prompt
    formatted_question = "Q: " + question
    formatted_question = formatted_question + "\nA: Let's work this out in a step by step way to be sure we have the right answer.\n"
    # Storing conversation elements
    conversation = []
    # Get response
    updated_convo = prompt_gpt_4_and_get_convo(conversation, formatted_question)
    return updated_convo

# Tree-of-Thought
def gpt4_gsm8k_tree_of_thought(question):
    # Format question
    formatted_question = "Q: " + question
    # Tree-of-Thoughts prompts
    tot_initial = formatted_question + "\nTask: Generate 3 different possible one-step calculations to serve as step 1 in solving the problem. Only work on step 1. Put each calculation on a new line. Do not number them."
    tot_prompt_2 = "Task: State the calculation above that is most likely to contribute to solving the problem. If it fully solves the original problem, also output STOP and the solution to the problem. If none of the calculations are correct, output ERROR and generate three new ones."
    tot_prompt_3 = "Task: Generate 3 different possible one-step calculations to serve as the next step in solving the problem. Only work on the next step. Put each calculation on a new line. Do not number them."
    # Storing conversation elements
    conversation = []
    updated_convo = []
    new_last_step = ""
    # Maximum of 16 items in the conversation
    for i in range(16):
        convo_to_feed = updated_convo.copy()
        last_step_this_loop = new_last_step
        # Get the response
        if i == 0:
            updated_convo = prompt_gpt_4_and_get_convo(conversation, tot_initial)
            # print('initial response')
            # print(updated_convo)
            new_last_step = "initial"
        # If the response contains STOP, stop
        if "STOP" in updated_convo[-1]['content']:
            # print('STOPPING')
            # print(updated_convo)
            return updated_convo
        # If last step was initial step or tot_prompt_3 need tot_prompt_2
        # Also run this step if the response from tot_prompt_2 contains ERROR
        if last_step_this_loop == "initial" or last_step_this_loop == "tot_prompt_3" or (last_step_this_loop == "tot_prompt_2" and "ERROR" in updated_convo[-1]['content']):
            updated_convo = prompt_gpt_4_and_get_convo(convo_to_feed, tot_prompt_2)
            # print('added tot2')
            # print(updated_convo)
            new_last_step = "tot_prompt_2"
        # If last step was tot_prompt_2 need tot_prompt_3
        if last_step_this_loop == "tot_prompt_2":
            updated_convo = prompt_gpt_4_and_get_convo(convo_to_feed, tot_prompt_3)
            # print('added tot3')
            # print(updated_convo)
            new_last_step = "tot_prompt_3"
    return updated_convo

# Self-Refine
def gpt4_gsm8k_self_refine(question):
    # Format question
    formatted_question = "Q: " + question
    formatted_question = formatted_question + "\nA:"
    # Self-Refine prompts
    self_refine_2 = "Task: Please check the answer above. If there is an error, state what the error is, but don't fix it. If there are no errors, output STOP.\nFeedback:"
    self_refine_3 = "Task: Redo the entire problem based on the most recent feedback.\nA:"
    # Self-refinement loop
    # Storing conversation elements
    conversation = []
    updated_convo = []
    new_last_step = ""
    # Maximum of 16 items in the conversation
    for i in range(16):
        convo_to_feed = updated_convo.copy()
        last_step_this_loop = new_last_step
        # Get the response
        if i == 0:
            updated_convo = prompt_gpt_4_and_get_convo(conversation, formatted_question)
            new_last_step = "initial"
        # If the response contains STOP, stop
        if "STOP" in updated_convo[-1]['content']:
            return updated_convo
        # If last step was initial step or self_refine_3 need self_refine_2
        if last_step_this_loop == "initial" or last_step_this_loop == "self_refine_3":
            updated_convo = prompt_gpt_4_and_get_convo(convo_to_feed, self_refine_2)
            new_last_step = "self_refine_2"
        # If last step was self_refine_2 need self_refine_3
        if last_step_this_loop == "self_refine_2":
            updated_convo = prompt_gpt_4_and_get_convo(convo_to_feed, self_refine_3)
            new_last_step = "self_refine_3"
    return updated_convo

# Least-to-Most
def gpt4_gsm8k_least_to_most(question):
    # Format question and prompt
    examples = """Q: Elsa has 5 apples. Anna has 2 more apples than Elsa. How many apples do they have together?\nA: Let's break down this problem: 1. How many apples does Anna have? 2. How many apples do Elsa and Anna have together?\n1. Anna has 2 more apples than Elsa. So Anna has 2 + 5 = 7 apples.2. Elsa and Anna have 5 + 7 = 12 apples together.\n"""
    formatted_question = examples + "Q: " + question
    formatted_question = formatted_question + "\nA: Let's break down this problem:\n"
    # Storing conversation elements
    conversation = []
    # Get response
    updated_convo = prompt_gpt_4_and_get_convo(conversation, formatted_question)
    return updated_convo

# Manual Few-Shot
def gpt4_gsm8k_manual_few_shot(question):
    # Format question and prompt
    examples = """Q: For every 12 cans you recycle, you receive $0.50, and for every 5 kilograms of newspapers, you receive $1.50. If your family collected 144 cans and 20 kilograms of newspapers, how much money would you receive?
    A: 12
    Q: Betty picked 16 strawberries. Matthew picked 20 more strawberries than Betty and twice as many as Natalie. They used their strawberries to make jam. One jar of jam used 7 strawberries and they sold each jar at $4. How much money were they able to make from the strawberries they picked?
    A: 40
    Q: Jack has a stack of books that is 12 inches thick. He knows from experience that 80 pages is one inch thick. If he has 6 books, how many pages is each one on average?
    A: 160
    Q: James dumps his whole collection of 500 Legos on the floor and starts building a castle out of them.  He uses half the pieces before finishing and is told to put the rest away.  He puts all of the leftover pieces back in the box they came from, except for 5 missing pieces that he can't find.  How many Legos are in the box at the end?
    A: 245
    Q: Ines had $20 in her purse. She bought 3 pounds of peaches, which are $2 per pound at the local farmers' market. How much did she have left?
    A: 14
    Q: Aaron pays his actuary membership fees each year. The membership fee increases yearly by $10. If he pays $80 in the first year, how much does his membership cost, in dollars, in the sixth year?
    A: 130
    Q: Joseph invested $1000 into a hedge fund. The fund promised a yearly interest rate of 10%. If he deposited an additional $100 every month into the account to add to his initial investment of $1000, how much money will he have in the fund after two years?
    A: 3982
    Q: The price of buying a wooden toy at the new Craftee And Best store is $20, and the cost of buying a hat is $10. If Kendra went to the shop with a $100 bill and bought two wooden toys and three hats, calculate the change she received.
    A: 30\n"""
    # Fix indentation
    examples = inspect.cleandoc(examples)
    formatted_question = examples + "Q: " + question
    formatted_question = formatted_question + "\nA:"
    # Storing conversation elements
    conversation = []
    # Get response
    updated_convo = prompt_gpt_4_and_get_convo(conversation, formatted_question)
    return updated_convo

# Manual Chain-of-Thought
def gpt4_gsm8k_manual_cot(question):
    # Format question and prompt
    examples = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
    A: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.
    Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
    A: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.
    Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
    A: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.
    Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
    A: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.
    Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
    A: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.
    Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
    A: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.
    Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
    A: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.
    Q: Olivia has \$23. She bought five bagels for \$3 each. How much money does she have left?
    A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.\n"""
    # Fix indentation
    examples = inspect.cleandoc(examples)
    formatted_question = examples + "Q: " + question
    formatted_question = formatted_question + "\nA:"
    # Storing conversation elements
    conversation = []
    # Get response
    updated_convo = prompt_gpt_4_and_get_convo(conversation, formatted_question)
    return updated_convo

####################################################################################################

# GPT-4
# Creative Writing

# Zero-Shot Control Baseline/Direct Prompting
def gpt4_cw_direct_prompting(sentences):
    # Create task
    task = "Write a coherent passage of 2 short paragraphs. The end sentence of each paragraph must be: " + " ".join(sentences)
    # Storing conversation elements
    conversation = []
    # Get response
    updated_convo = prompt_gpt_4_and_get_convo(conversation, task)
    return updated_convo

# Zero-shot Chain-of-Thought
def gpt4_cw_zero_shot_cot(sentences):
    # Create task
    task = "Write a coherent passage of 2 short paragraphs. The end sentence of each paragraph must be: " + " ".join(sentences)
    # Format task by adding to it
    formatted_task = task + " Plan step-by-step before writing the passage."
    # Storing conversation elements
    conversation = []
    # Get response
    updated_convo = prompt_gpt_4_and_get_convo(conversation, formatted_task)
    return updated_convo

# APE Improved Zero-Shot Chain-of-Thought
def gpt4_cw_ape_zero_shot_cot(sentences):
    # Create task
    task = "Write a coherent passage of 2 short paragraphs. The end sentence of each paragraph must be: " + " ".join(sentences)
    # Format task by adding to it
    formatted_task = task + " Plan step-by-step before writing the passage to be sure we have a correct and coherent answer."
    # Storing conversation elements
    conversation = []
    # Get response
    updated_convo = prompt_gpt_4_and_get_convo(conversation, formatted_task)
    return updated_convo

# Tree-of-Thought
def gpt4_cw_tree_of_thought(sentences):
    # Create task - note that this is not the same as other methods
    task = "Goal: A coherent passage of 2 short paragraphs. The end sentence of each paragraph, respectively, must be: " + " ".join(sentences)
    # Tree-of-Thoughts prompts
    tot_initial = "Your Task: Generate 3 one-sentence plans for potential passages. Only generate one-sentence plans - do not write the passage."
    tot_prompt_2 = "Your Task: Select the most coherent plan that follows the rules of the task. Only state the plan - do not write the passage."
    tot_prompt_3 = "Your Task: Write 3 drafts of the 2-paragraph passage based on this plan."
    tot_prompt_4 = "Your Task: Select the most coherent draft that follows the rules of the task and write it out."
    tot_prompt_5 = "Your Task: If the draft is correct and coherent to the extent you would award a 10 on a scale of 1 to 10, output STOP. If it is not, write out a different one-sentence plan for a potential passage from among those considered and output PLAN."
    # Begin conversation
    conversation = []
    updated_convo = []
    initial_prompt = task + "\n" + tot_initial
    last_step = ""
    new_last_step = ""
    # Maximum of 16 items in the conversation
    for i in range(16):
        last_step = new_last_step
        # Initial prompting
        if i == 0:
            # Get the response
            updated_convo = prompt_gpt_4_and_get_convo(conversation, initial_prompt)
            new_last_step = "initial"
        # If the response contains STOP, stop
        if "STOP" in updated_convo[-1]['content']:
            # print('STOPPING')
            # print(updated_convo)
            return updated_convo
        # If last step was initial step need tot_prompt_2
        if last_step == "initial":
            updated_convo = prompt_gpt_4_and_get_convo(updated_convo, tot_prompt_2)
            # print('added tot2')
            # print(updated_convo)
            new_last_step = "tot_prompt_2"
        # If last step was tot_prompt_2 need tot_prompt_3
        # Also return to this step if the response from tot_prompt_5 contains PLAN
        if last_step == "tot_prompt_2" or (last_step == "tot_prompt_5" and "PLAN" in updated_convo[-1]['content']):
            updated_convo = prompt_gpt_4_and_get_convo(updated_convo, tot_prompt_3)
            # print('added tot3')
            # print(updated_convo)
            new_last_step = "tot_prompt_3"
        # If last step was tot_prompt_3 need tot_prompt_4
        if last_step == "tot_prompt_3":
            updated_convo = prompt_gpt_4_and_get_convo(updated_convo, tot_prompt_4)
            # print('added tot4')
            # print(updated_convo)
            new_last_step = "tot_prompt_4"
        # If last step was tot_prompt_4 need tot_prompt_5
        if last_step == "tot_prompt_4":
            updated_convo = prompt_gpt_4_and_get_convo(updated_convo, tot_prompt_5)
            # print('added tot5')
            # print(updated_convo)
            new_last_step = "tot_prompt_5"
    # print(updated_convo)
    return updated_convo

# Self-Refine
def gpt4_cw_self_refine(sentences):
    # Create task
    task = "Task: Write a coherent passage of 2 short paragraphs. The end sentence of each paragraph must be: " + " ".join(sentences) + "\nResponse: "
    # Self-Refine prompts
    self_refine_2 = "Your Task: Provide feedback on the correctness and coherence of the response and a rating on a scale of 1-10. If it is already coherent and correct to the extent you would award a 10, output 10 and the word STOP.\nResponse: "
    self_refine_3 = "Your Task: Rewrite the passage based on the most recent feedback.\nResponse: "
    # Self-refinement loop
    # Storing conversation elements
    conversation = []
    updated_convo = []
    new_last_step = ""
    # Maximum of 16 items in the conversation
    for i in range(16):
        convo_to_feed = updated_convo.copy()
        last_step_this_loop = new_last_step
        # Get the response
        if i == 0:
            updated_convo = prompt_gpt_4_and_get_convo(conversation, task)
            new_last_step = "initial"
        # If the response contains STOP, stop
        if "STOP" in updated_convo[-1]['content']:
            return updated_convo
        # If last step was initial step or self_refine_3 need self_refine_2
        if last_step_this_loop == "initial" or last_step_this_loop == "self_refine_3":
            updated_convo = prompt_gpt_4_and_get_convo(convo_to_feed, self_refine_2)
            new_last_step = "self_refine_2"
        # If last step was self_refine_2 need self_refine_3
        if last_step_this_loop == "self_refine_2":
            updated_convo = prompt_gpt_4_and_get_convo(convo_to_feed, self_refine_3)
            new_last_step = "self_refine_3"
    return updated_convo

# Least-to-Most
def gpt4_cw_least_to_most(sentences):
    # Create examples
    examples = """Task: Write a coherent passage of 2 short paragraphs. The end sentence of each paragraph, respectively, must be: 1. She couldn't get a job because she didn't have a permanent address. 2. He didn't have any hidden talents.
    Response: Let's break down this problem into steps: First, we will write a few ideas for the passage. Second, draft the passage.
    Ideas:
    1. Make the passage about travelling circus performers looking for other work as their circus shuts down.
    2. Make the passage about the struggles of a homeless person who is trying to get a job. 
    3. Make the passage about perceptions and preconceptions of people's skills and social status as factors in hiring.
    Passage: 
    Laura sat on the park bench, watching the people walk by. She was homeless, and had been for a few months now. She couldn't get a job because she didn't have a permanent address.
    She had tried to talk to career counselors about her situation, but the conversations often seemed fruitless. She didn't feel she had any marketable skills. Her situation was similar to that of her friend, Rodrigo, who openly shared a similar attitude with counselors in his meetings. He didn't have any hidden talents.
    Task: Write a coherent passage of 2 short paragraphs. The end sentence of each paragraph, respectively, must be: 1. He had always wanted to be a Youtuber but never thought it would actually happen. 2. My sweater got caught on the door hinge.
    Response: Let's break down this problem into steps: First, we will write a few ideas for the passage. Second, draft the passage.
    Ideas:
    1. Make the passage about a sister visiting her brother; the brother has recently become a successful Youtuber - she excitedly gets her sweater caught leaving a meeting with him.
    2. Make the passage about a men's fashion reviewer who is working on a video review of a sweater.
    3. Make the passage about a Youtuber preparing for a video shoot - as they hurry through things, their sweater gets caught but this becomes an amusing part of their vlog.
    Passage:
    My brother, John, had been making home videos for years, but they never got much attention. He was always disappointed when he saw other people's videos getting thousands of views. Then one day, he got a call from a company that wanted to sponsor him. They offered him a lot of money to make videos for them. He was so excited that he couldn't sleep that night. He had always wanted to be a Youtuber but never thought it would actually happen.
    As it turned out, John would need his own production staff to help with script writing and video editing. As I lived in the area and had prior experience in these fields, I was a natural choice for a part-time role on his channel. The company's sponsorship was very generous, and I would get a large portion of the profits. I was glad to finally be able to earn a substantial income in a more exciting and engaging role than my current position as a barista. I was smiling for most of our first business meeting, and strutted with pride out of our new studio. My sweater got caught on the door hinge."""
    # Fix indentation
    examples = inspect.cleandoc(examples)
    # Create task
    task = examples + "\nTask: Write a coherent passage of 2 short paragraphs. The end sentence of each paragraph must be: " + " ".join(sentences) + "\nResponse: "
    # Storing conversation elements
    conversation = []
    # Get response
    updated_convo = prompt_gpt_4_and_get_convo(conversation, task)
    return updated_convo

# Manual Few-Shot
def gpt4_cw_manual_few_shot(sentences):
    # Create examples
    examples = """Task: Write a coherent passage of 2 short paragraphs. The end sentence of each paragraph, respectively, must be: 1. She couldn't get a job because she didn't have a permanent address. 2. He didn't have any hidden talents.
    Response: 
    Laura sat on the park bench, watching the people walk by. She was homeless, and had been for a few months now. She couldn't get a job because she didn't have a permanent address.
    She had tried to talk to career counselors about her situation, but the conversations often seemed fruitless. She didn't feel she had any marketable skills. Her situation was similar to that of her friend, Rodrigo, who openly shared a similar attitude with counselors in his meetings. He didn't have any hidden talents.
    Task: Write a coherent passage of 2 short paragraphs. The end sentence of each paragraph, respectively, must be: 1. He had always wanted to be a Youtuber but never thought it would actually happen. 2. My sweater got caught on the door hinge.
    Response: 
    My brother, John, had been making home videos for years, but they never got much attention. He was always disappointed when he saw other people's videos getting thousands of views. Then one day, he got a call from a company that wanted to sponsor him. They offered him a lot of money to make videos for them. He was so excited that he couldn't sleep that night. He had always wanted to be a Youtuber but never thought it would actually happen.
    As it turned out, John would need his own production staff to help with script writing and video editing. As I lived in the area and had prior experience in these fields, I was a natural choice for a part-time role on his channel. The company's sponsorship was very generous, and I would get a large portion of the profits. I was glad to finally be able to earn a substantial income in a more exciting and engaging role than my current position as a barista. I was smiling for most of our first business meeting, and strutted with pride out of our new studio. My sweater got caught on the door hinge."""
    # Fix indentation
    examples = inspect.cleandoc(examples)
    # Create task
    task = examples + "\nTask: Write a coherent passage of 2 short paragraphs. The end sentence of each paragraph must be: " + " ".join(sentences) + "\nResponse: "
    # Storing conversation elements
    conversation = []
    # Get response
    updated_convo = prompt_gpt_4_and_get_convo(conversation, task)
    return updated_convo

# Manual Chain-of-Thought
def gpt4_cw_manual_cot(sentences):
    # Create examples
    examples = """Task: Write a coherent passage of 2 short paragraphs. The end sentence of each paragraph, respectively, must be: 1. She couldn't get a job because she didn't have a permanent address. 2. He didn't have any hidden talents.
    Response: 
    Ideas:
    1. Make the passage about travelling circus performers looking for other work as their circus shuts down.
    2. Make the passage about the struggles of a homeless person who is trying to get a job. 
    3. Make the passage about perceptions and preconceptions of people's skills and social status as factors in hiring.
    Passage: 
    Laura sat on the park bench, watching the people walk by. She was homeless, and had been for a few months now. She couldn't get a job because she didn't have a permanent address.
    She had tried to talk to career counselors about her situation, but the conversations often seemed fruitless. She didn't feel she had any marketable skills. Her situation was similar to that of her friend, Rodrigo, who openly shared a similar attitude with counselors in his meetings. He didn't have any hidden talents.
    Task: Write a coherent passage of 2 short paragraphs. The end sentence of each paragraph, respectively, must be: 1. He had always wanted to be a Youtuber but never thought it would actually happen. 2. My sweater got caught on the door hinge.
    Response:
    Ideas:
    1. Make the passage about a sister visiting her brother; the brother has recently become a successful Youtuber - she excitedly gets her sweater caught leaving a meeting with him.
    2. Make the passage about a men's fashion reviewer who is working on a video review of a sweater.
    3. Make the passage about a Youtuber preparing for a video shoot - as they hurry through things, their sweater gets caught but this becomes an amusing part of their vlog.
    Passage:
    My brother, John, had been making home videos for years, but they never got much attention. He was always disappointed when he saw other people's videos getting thousands of views. Then one day, he got a call from a company that wanted to sponsor him. They offered him a lot of money to make videos for them. He was so excited that he couldn't sleep that night. He had always wanted to be a Youtuber but never thought it would actually happen.
    As it turned out, John would need his own production staff to help with script writing and video editing. As I lived in the area and had prior experience in these fields, I was a natural choice for a part-time role on his channel. The company's sponsorship was very generous, and I would get a large portion of the profits. I was glad to finally be able to earn a substantial income in a more exciting and engaging role than my current position as a barista. I was smiling for most of our first business meeting, and strutted with pride out of our new studio. My sweater got caught on the door hinge."""
    # Fix indentation
    examples = inspect.cleandoc(examples)
    # Create task
    task = examples + "\nTask: Write a coherent passage of 2 short paragraphs. The end sentence of each paragraph must be: " + " ".join(sentences) + "\nResponse: "
    # Storing conversation elements
    conversation = []
    # Get response
    updated_convo = prompt_gpt_4_and_get_convo(conversation, task)
    return updated_convo
