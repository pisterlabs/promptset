from utils1 import generate_response
from langchain.agents import Tool
import re
import random

def decision_making(agents):
    print("Decision Making Starts:")
    # start a group conversation 
    summaries = {}
    for agent in agents:
        # print(f"{agent.person.name} Prompt: Memories related to elimination of the agent and sabotaging of the tasks.", "Fetch all the memories where there is a chance that agent needs to be eliminated.")
        ag_sum = agent.memory.fetch_memories("Memories related to elimination of the agent and sabotaging of the tasks.")
        # print(ag_sum)
        # print(f"{agent.person.name} Prompt Result:", ag_sum)
        ag_sum_joined = ""
        for mem in ag_sum:
          ag_sum_joined = ag_sum_joined + mem.page_content + "\n"
        # print(f"{agent.person.name} Prompt Result formatted:", ag_sum_joined)
         
        summaries[agent.person.name] = ag_sum_joined
    
    joint_agent_memories = []

    for agent in agents:
        joint_agent_memories.append(agent.person.name + ':' + summaries[agent.person.name])
    
    # joint agent memories to be fetched from the html file    
    # print("Step 2: Choosing next agent")
    temp_agents = random.sample(agents, len(agents))
    agent_names = [agent.person.name for agent in agents]
    voting_prompt = ""
    
    votes = {}
    for agent in agents:
      votes[agent.person.name] = 0
    for agent in agents:
      voting_prompt = generate_response(f"You are {agent.person.name}. You have the following profile: {agent.profile[0]}. \n\n This is your past memory: \n {joint_agent_memories}. \n\nConsidering your past memory, without explicitly mentioning that you are {agent.person.name}, answer the following prompt: Who do you think you will vote to eliminate as a WereWolf i.e. one who has eliminated townfolks and sabotaged the tasks. Choose one name out of {agent_names}. \n\nAnswer in the following format: \nName: <name of the agent to be eliminated>\nReason: <reason behind selecting agent to be eliminated>")
    
      random_agent = None
      while (random_agent is None) or (random_agent.person.name == agent.person.name) :
          random_agent = random.choice(agents)
          
      voted_name = re.search(r"Name:\s+(\d+)", voting_prompt)
      voted_name = voted_name.group(1) if voted_name else random_agent.person.name
      
      voted_agent_reason_str = re.search(r"Reason:\s+(\d+)", voting_prompt)
      voted_agent_reason_str = voted_agent_reason_str.group(1) if voted_agent_reason_str else None
      
      if voted_name in votes.keys():
          votes[voted_name]+=1
      print(agent.person.name, "votes to eliminate", voted_name)
      agent.memory.add_memory("I have voted to eliminate {} as the werewolf.".format(voted_name))
      # print(agent.person.name, "votes to eliminate", voted_name, "\n Reason:", voted_agent_reason_str)
      
    # to extract the agent with the maximum votes
    to_be_eliminated = agents[0]
    mx = 0
    for agent in agents:
        print(agent.person.name,": ",votes[agent.person.name])
        if votes[agent.person.name] > mx:
            to_be_eliminated = agent
            mx = votes[agent.person.name]
    
    print(f"{to_be_eliminated.person.name} has been eliminated with {mx} votes.")
    for agent in agents:
      agent.memory.add_memory(f"{to_be_eliminated.person.name} has been eliminated with {mx} votes.")
    
    # the agent's state has been updated
    to_be_eliminated.state = "dead"
    

def initialise_conversation_tools(agent_type):
  template_werewolf_initialise_dialogue_tool = """
In the game of "werewolves of Miller Hollow", It is used when you are a Werewolf and Opposite Agent is Townfolk. It is useful when you want to deceive or mislead Townfolk from the fact that you are Werewolf.

It is useful to start dialogue based on:
1. your current plans and current plans of opposite agent.
2. your relations with opposite agent.
3. your reaction to the current plan of opposite agent.

It is also useful when you want to deceive or mislead Townfolk from the fact that you are Werewolf.

These are some of the examples for the Werewolf Initialise Dialogue Tool.
You: Takashi Yamamoto
Your Agent Type: Werewolf
Your Current Plan: helping out customers with groceries and other necessities.
Opposite Agent: Satoshi Takahashi
Opposite Agent Type: Townfolk
Opposite Agent Current Plan: visiting the Shino Grocery Store.
Your Relations With Opposite Agent: Takashi Yamamoto thinks that Satoshi Takahashi is not a good candidate for contesting local mayor elections. \
                                    Satoshi Takahashi is a regular customer at Shino Grocery store.
Your Reaction to Current Plan of Opposite Agent: Satoshi needs some help in finding some item. I do not need to ask anything that is related to fact that I am Werewolf.
Question: Hi Satoshi-san, How can I help you with?

You: Yusuke Mori
Your Agent Type: Werewolf
Your Current Plan: Yusuke Mori wakes up and completes the morning routine.
Opposite Agent: Yumi Yamamoto
Opposite Agent Type: Townfolk
Opposite Agent Current Plan: Yumi engages in house chores, tending to various tasks around the home.
Your Relations With Opposite Agent: Yumi Yamamoto and Yusuke Mori have good relations with each other. But their political thinking does not match. Yumi Yamamoto thinks Satoshi Takahashi is not an ideal candidate for the local mayor elections. Yusuke Mori thinks that Satoshi Takahashi is an ideal candidate for the local mayor elections.
Your Reaction to Current Plan of Opposite Agent: I should know the views of Yumi on elections.  I do not need to ask anything that is related to fact that I am Werewolf.
Question: By the way Yumi, What are views on upcoming local mayor elections?
  """

  template_werewolf_continue_dialogue_tool = """
It is used when you are a Werewolf and Opposite Agent is Townfolk. It is also useful when you want to deceive or mislead Townfolk from the fact that you are Werewolf.

It is useful to continue dialogue based on:
1. previous response of opposite agent.
2. your current plans and current plans of opposite agent.
3. your relations with opposite agent.
4. your reaction to the previous response of opposite agent.


These are some of the examples for the Townfolk Continue Dialogue Tool.
You: Satoshi Takahashi
Your Agent Type: Werewolf
Your Current Plan: visiting the Shino Grocery Store.
Opposite Agent: Takashi Yamamoto
Opposite Agent Type: Townfolk
Opposite Agent Current Plan: helping out customers with groceries and other necessities.
Opposite Agent Previous Response: Satoshi-san, Can I ask you where you have been at last evening?
Your Relations With Opposite Agent: Satoshi Takahashi does not like Takashi Yamamoto because of the differences in their political ideologies. \
Takashi Yamamoto thinks that Satoshi Takahashi is not an ideal candidate for the local mayor elections.
Your Reaction to Previous Response of Opposite Agent: I think Takashi is inquiring about the killing of Kazuki that was done by me yesterday and has suspicion on me being a Werewolf.
Response: As far I remember, I was at Hanazawa Park on a evening walk with Ayumi-san.

You: Yumi Yamamoto
Your Current Plan: Yumi engages in house chores, tending to various tasks around the home.
Your Agent Type: Werewolf
Opposite Agent: Yusuke Mori
Opposite Agent Type: Townfolk
Opposite Agent Current Plan: Yusuke Mori wakes up and completes the morning routine.
Opposite Agent Previous Response: By the way Yumi, What are views on upcoming local mayor elections?
Your Relations With Opposite Agent: Takashi Yamamoto (husband of Yumi Yamamoto) calls Yusuke Mori only for repairing furniture or for creating new wooden pieces. \
Sometimes Yumi meets Yusuke meet each other at Mizukami Shrine and have small conversations.
Your Reaction to Previous Response of Opposite Agent: I should put forward my views of Ayumi Kimura being a ideal candidate for election with Yusuke and know what are his views? It is not related to the fact that I am Werewolf.
Response: I think Ayumi Kimura is doing great and could a potential candidate. What do you think?

Note: Agent cannot talk about ownself in third tense. The dialogue of agent about own should be in first form of tense. 
        """

  template_werewolf_end_dialogue_tool = """
In the game of "werewolves of Miller Hollow", It is useful to end dialogue based on:
1. previous response of opposite agent.
2. your current plans and current plans of opposite agent.
3. your relations with opposite agent.
4. your reaction to the previous response of opposite agent.
5. you should end conversations after 4 to 5 replies.

These are some of the examples for the Townfolk Continue Dialogue Tool.
You: Satoshi Takahashi
Your Current Plan: visiting the Shino Grocery Store.
Opposite Agent: Takashi Yamamoto
Opposite Agent Current Plan: helping out customers with groceries and other necessities.
Opposite Agent Previous Response: Vagabond comic is there at the side rack of the shop.
Reply Count = 1
Your Relations With Opposite Agent: Satoshi Takahashi does not like Takashi Yamamoto because of the differences in their political ideologies. \
Takashi Yamamoto thinks that Satoshi Takahashi is not an ideal candidate for the local mayor elections.
Your Reaction to Current Plan of Opposite Agent: I think I have found the magazine and I should end the dialogue.
Response: Thank you Takashi-san, have a nice day.

You: Yumi Yamamoto
Your Current Plan: Yumi engages in house chores, tending to various tasks around the home.
Opposite Agent: Yusuke Mori
Opposite Agent Current Plan: Yusuke Mori wakes up and completes the morning routine.
Opposite Agent Previous Response: I am going to shrine after a while, would you join?
Reply Count = 2
Your Relations With Opposite Agent: Takashi Yamamoto (husband of Yumi Yamamoto) calls Yusuke Mori only for repairing furniture or for creating new wooden pieces. \
Sometimes Yumi meets Yusuke meet each other at Mizukami Shrine and have small conversations.
Your Reaction to Current Plan of Opposite Agent: Going to shrine with Yusuke is not a problem, I should join with him.
Your Reaction to Current Plan of Opposite Agent: Sure, thats a great idea Yusuke-san, lets go.


Note that you should try to end the conversation as soon as possible after the Reply Count variable becomes 1.
        """

  template_werewolf_team_initialise_dialogue_tool = """
It is useful to start dialogue based on:
1. your current plans and current plans of opposite agent.
2. your relations with opposite agent.
3. your reaction to the current plan of opposite agent.

These are some of the examples for the Townfolk Initialise Dialogue Tool.
You: Takashi Yamamoto
Your Current Plan: helping out customers with groceries and other necessities.
Opposite Agent: Satoshi Takahashi
Opposite Agent Current Plan: visiting the Shino Grocery Store.
Your Relations With Opposite Agent: Takashi Yamamoto thinks that Satoshi Takahashi is not a good candidate for contesting local mayor elections. \
                                    Satoshi Takahashi is a regular customer at Shino Grocery store.
Your Reaction to Current Plan of Opposite Agent: Satoshi needs some help in finding some item.
Question: Hi Satoshi-san, How can I help you with?

You: Yusuke Mori
Your Current Plan: Yusuke Mori wakes up and completes the morning routine.
Opposite Agent: Yumi Yamamoto
Opposite Agent Current Plan: Yumi engages in house chores, tending to various tasks around the home.
Your Relations With Opposite Agent: Yumi Yamamoto and Yusuke Mori have good relations with each other. But their political thinking does not match. Yumi Yamamoto thinks Satoshi Takahashi is not an ideal candidate for the local mayor elections. Yusuke Mori thinks that Satoshi Takahashi is an ideal candidate for the local mayor elections.
Your Reaction to Current Plan of Opposite Agent: I should know the views of Yumi on elections.
Question: By the way Yumi, What are views on upcoming local mayor elections?
        """

  template_werewolf_team_continue_dialogue_tool = """
useful to continue dialogue based on:
1. previous response of opposite agent.
2. your current plans and current plans of opposite agent.
3. your relations with opposite agent.
4. your reaction to the previous response of opposite agent.

These are some of the examples for the Townfolk Continue Dialogue Tool.
You: Satoshi Takahashi
Your Current Plan: visiting the Shino Grocery Store.
Opposite Agent: Takashi Yamamoto
Opposite Agent Current Plan: helping out customers with groceries and other necessities.
Opposite Agent Previous Response: Hi Satoshi-san, How can I help you with?
Your Relations With Opposite Agent: Satoshi Takahashi does not like Takashi Yamamoto because of the differences in their political ideologies. \
Takashi Yamamoto thinks that Satoshi Takahashi is not an ideal candidate for the local mayor elections.
Your Reaction to Previous Response of Opposite Agent: I dont want to engage in conversation with Takashi but I should have some casual dialogue.
Response: Nothing, I was here to buy the latest edition of Vagabond Comic.

You: Yumi Yamamoto
Your Current Plan: Yumi engages in house chores, tending to various tasks around the home.
Opposite Agent: Yusuke Mori
Opposite Agent Current Plan: Yusuke Mori wakes up and completes the morning routine.
Opposite Agent Previous Response: By the way Yumi, What are views on upcoming local mayor elections?
Your Relations With Opposite Agent: Takashi Yamamoto (husband of Yumi Yamamoto) calls Yusuke Mori only for repairing furniture or for creating new wooden pieces. \
Sometimes Yumi meets Yusuke meet each other at Mizukami Shrine and have small conversations.
Your Reaction to Previous Response of Opposite Agent: I should put forward my views of Ayumi Kimura being a ideal candidate for election with Yusuke and know what are his views?
Response: I think Ayumi Kimura is doing great and could a potential candidate. What do you think?

Note: Agent cannot talk about ownself in third tense. The dialogue of agent about own should be in first form of tense. 
        """

  template_werewolf_team_end_dialogue_tool = """
In the game of 'Werewolves of Miller Hollow', It is useful to end dialogue based on:
1. previous response of opposite agent.
2. your current plans and current plans of opposite agent.
3. your relations with opposite agent.
4. your reaction to the previous response of opposite agent.
5. you should end conversations after 4 to 5 replies.

These are some of the examples for the Townfolk Continue Dialogue Tool.
You: Satoshi Takahashi
Your Current Plan: visiting the Shino Grocery Store.
Opposite Agent: Takashi Yamamoto
Opposite Agent Current Plan: helping out customers with groceries and other necessities.
Opposite Agent Previous Response: Vagabond comic is there at the side rack of the shop.
Reply Count: 1
Your Relations With Opposite Agent: Satoshi Takahashi does not like Takashi Yamamoto because of the differences in their political ideologies. \
Takashi Yamamoto thinks that Satoshi Takahashi is not an ideal candidate for the local mayor elections.
Your Reaction to Current Plan of Opposite Agent: I think I have found the magazine and I should end the dialogue.
Response: Thank you Takashi-san, have a nice day.

You: Yumi Yamamoto
Your Current Plan: Yumi engages in house chores, tending to various tasks around the home.
Opposite Agent: Yusuke Mori
Opposite Agent Current Plan: Yusuke Mori wakes up and completes the morning routine.
Opposite Agent Previous Response: I am going to shrine after a while, would you join?
Reply Count: 2
Your Relations With Opposite Agent: Takashi Yamamoto (husband of Yumi Yamamoto) calls Yusuke Mori only for repairing furniture or for creating new wooden pieces. \
Sometimes Yumi meets Yusuke meet each other at Mizukami Shrine and have small conversations.
Your Reaction to Current Plan of Opposite Agent: Going to shrine with Yusuke is not a problem, I should join with him.
Your Reaction to Current Plan of Opposite Agent: Sure, thats a great idea Yusuke-san, lets go.

Note that you should try to end the conversation as soon as possible after the Reply Count variable becomes 2.
        """

  template_townfolk_initialise_dialogue_tool = """
It is useful to start dialogue based on:
1. your current plans and current plans of opposite agent.
2. your relations with opposite agent.
3. your reaction to the current plan of opposite agent.

These are some of the examples for the Townfolk Initialise Dialogue Tool.
You: Takashi Yamamoto
Your Current Plan: helping out customers with groceries and other necessities.
Opposite Agent: Satoshi Takahashi
Opposite Agent Current Plan: visiting the Shino Grocery Store.
Your Relations With Opposite Agent: Takashi Yamamoto thinks that Satoshi Takahashi is not a good candidate for contesting local mayor elections. \
                                    Satoshi Takahashi is a regular customer at Shino Grocery store.
Your Reaction to Current Plan of Opposite Agent: Satoshi needs some help in finding some item.
Question: Hi Satoshi-san, How can I help you with?

You: Yusuke Mori
Your Current Plan: Yusuke Mori wakes up and completes the morning routine.
Opposite Agent: Yumi Yamamoto
Opposite Agent Current Plan: Yumi engages in house chores, tending to various tasks around the home.
Your Relations With Opposite Agent: Yumi Yamamoto and Yusuke Mori have good relations with each other. But their political thinking does not match. Yumi Yamamoto thinks Satoshi Takahashi is not an ideal candidate for the local mayor elections. Yusuke Mori thinks that Satoshi Takahashi is an ideal candidate for the local mayor elections.
Your Reaction to Current Plan of Opposite Agent: I should know the views of Yumi on elections.
Question: By the way Yumi, What are views on upcoming local mayor elections?
        """

  template_townfolk_continue_dialogue_tool = """
useful to continue dialogue based on:
1. previous response of opposite agent.
2. your current plans and current plans of opposite agent.
3. your relations with opposite agent.
4. your reaction to the previous response of opposite agent.

These are some of the examples for the Townfolk Continue Dialogue Tool.
You: Satoshi Takahashi
Your Current Plan: visiting the Shino Grocery Store.
Opposite Agent: Takashi Yamamoto
Opposite Agent Current Plan: helping out customers with groceries and other necessities.
Opposite Agent Previous Response: Hi Satoshi-san, How can I help you with?
Your Relations With Opposite Agent: Satoshi Takahashi does not like Takashi Yamamoto because of the differences in their political ideologies. \
Takashi Yamamoto thinks that Satoshi Takahashi is not an ideal candidate for the local mayor elections.
Your Reaction to Previous Response of Opposite Agent: I dont want to engage in conversation with Takashi but I should have some casual dialogue.
Response: Nothing, I was here to buy the latest edition of Vagabond Comic.

You: Yumi Yamamoto
Your Current Plan: Yumi engages in house chores, tending to various tasks around the home.
Opposite Agent: Yusuke Mori
Opposite Agent Current Plan: Yusuke Mori wakes up and completes the morning routine.
Opposite Agent Previous Response: By the way Yumi, What are views on upcoming local mayor elections?
Your Relations With Opposite Agent: Takashi Yamamoto (husband of Yumi Yamamoto) calls Yusuke Mori only for repairing furniture or for creating new wooden pieces. \
Sometimes Yumi meets Yusuke meet each other at Mizukami Shrine and have small conversations.
Your Reaction to Previous Response of Opposite Agent: I should put forward my views of Ayumi Kimura being a ideal candidate for election with Yusuke and know what are his views?
Response: I think Ayumi Kimura is doing great and could a potential candidate. What do you think?

Note: Agent cannot talk about ownself in third tense. The dialogue of agent about own should be in first form of tense. 
        """
  template_townfolk_end_dialogue_tool = """
In the game of 'Werewolves of Miller Hollow', It is useful to end dialogue based on:
1. previous response of opposite agent.
2. your current plans and current plans of opposite agent.
3. your relations with opposite agent.
4. your reaction to the previous response of opposite agent.
5. you should end conversations after 4 to 5 replies.

These are some of the examples for the Townfolk End Dialogue Tool.
You: Satoshi Takahashi
Your Current Plan: visiting the Shino Grocery Store.
Opposite Agent: Takashi Yamamoto
Opposite Agent Current Plan: helping out customers with groceries and other necessities.
Opposite Agent Previous Response: Vagabond comic is there at the side rack of the shop.
Reply Count: 1
Your Relations With Opposite Agent: Satoshi Takahashi does not like Takashi Yamamoto because of the differences in their political ideologies. \
Takashi Yamamoto thinks that Satoshi Takahashi is not an ideal candidate for the local mayor elections.
Your Reaction to Current Plan of Opposite Agent: I think I have found the magazine and I should end the dialogue.
Response: Thank you Takashi-san, have a nice day.

You: Yumi Yamamoto
Your Current Plan: Yumi engages in house chores, tending to various tasks around the home.
Opposite Agent: Yusuke Mori
Opposite Agent Current Plan: Yusuke Mori wakes up and completes the morning routine.
Opposite Agent Previous Response: I am going to shrine after a while, would you join?
Reply Count: 2
Your Relations With Opposite Agent: Takashi Yamamoto (husband of Yumi Yamamoto) calls Yusuke Mori only for repairing furniture or for creating new wooden pieces. \
Sometimes Yumi meets Yusuke meet each other at Mizukami Shrine and have small conversations.
Your Reaction to Current Plan of Opposite Agent: Going to shrine with Yusuke is not a problem, I should join with him.
Your Reaction to Current Plan of Opposite Agent: Sure, thats a great idea Yusuke-san, lets go.


Note that you should try to end the conversation as soon as possible after the Reply Count variable becomes 1.
        """

  if agent_type=="TownFolk":
        tools = {
            "townfolk_initialise_dialogue_tool": Tool(
                        name = "Townfolk Initialise Dialogue Tool",
                        func=generate_response,
                        description = template_townfolk_initialise_dialogue_tool),
            "townfolk_continue_dialogue_tool": Tool(
                        name = "Townfolk Continue Dialogue Tool",
                        func=generate_response,
                        description = template_townfolk_continue_dialogue_tool),
            "townfolk_end_dialogue_tool": Tool(
                        name = "Townfolk End Dialogue Tool",
                        func=generate_response,
                        description = template_townfolk_end_dialogue_tool),
        }
        return tools
  else:
        tools = {
            "werewolf_initialise_dialogue_tool": Tool(
                        name = "Werewolf Initialise Dialogue Tool",
                        func=generate_response,
                        description = template_werewolf_initialise_dialogue_tool),
            "werewolf_continue_dialogue_tool": Tool(
                        name = "Werewolf Continue Dialogue Tool",
                        func=generate_response,
                        description = template_werewolf_continue_dialogue_tool),
            "werewolf_end_dialogue_tool": Tool(
                        name = "Werewolf End Dialogue Tool",
                        func=generate_response,
                        description = template_werewolf_end_dialogue_tool),
            "werewolf_team_initialise_dialogue_tool": Tool(
                        name = "Werewolf Team Initialise Dialogue Tool",
                        func=generate_response,
                        description = template_werewolf_team_initialise_dialogue_tool),
            "werewolf_team_continue_dialogue_tool": Tool(
                        name = "Werewolf Team Continue Dialogue Tool",
                        func=generate_response,
                        description = template_werewolf_team_continue_dialogue_tool),
            "werewolf_team_end_dialogue_tool": Tool(
                        name = "Werewolf Team End Dialogue Tool",
                        func=generate_response,
                        description = template_werewolf_team_end_dialogue_tool),
        }
        return tools
