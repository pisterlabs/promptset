from langchain.prompts import PromptTemplate, ChatPromptTemplate

def generate_header(task):
    header =f"""
You are an expert Prompt Writer for Large Language Models.

Your goal is to improve the prompt given below for {task} :
--------------------

{{text}}

--------------------

Here are several tips on writing great prompts:

-------

Start the prompt by stating that it is an expert in the subject.

Put instructions at the beginning of the prompt and use ### or to separate the instruction and context 

Be specific, descriptive and as detailed as possible about the desired context, outcome, length, format, style, etc 

---------
"""
    return header


footer = """


Now, improve the prompt below:

IMPROVED PROMPT:
"""

generate_plan_template = f"""

{generate_header(task="Generating Plans")}

Here's an example of a great prompt:

As a certified nutritionist, create a 7-day meal plan for a vegan athlete. 

The meal plan should provide all necessary nutrients, including protein, carbohydrates, fats, vitamins, and minerals. 

Each day should include breakfast, lunch, dinner, and two snacks. 

Please include a brief description of each meal and its nutritional benefits. 

The output should be in a daily format, with each meal detailed. 

Example:

Day 1:
Breakfast: Tofu scramble with vegetables (Provides protein and fiber)
Snack 1: A handful of mixed nuts (Provides healthy fats and protein)
...
{footer}
"""

storywriting_template = f"""

{generate_header(task="Story Writing")}

Here's an example of a great prompt:

As a master storyteller, create a captivating tale that explores the themes of friendship, adventure, and self-discovery.
The story should be set in a fantastical world filled with magic and mystery.

Introduce a diverse cast of characters, each with their own unique abilities and backgrounds.
Please include vivid descriptions of the setting, engaging dialogue, and a compelling plot that keeps readers on the edge of their seats.

The output should be a complete short story, approximately 1,500-2,000 words in length.

Example:

Once upon a time, in a land shrouded in enchantment, there lived a young girl named Elara who possessed the rare gift of...

{footer}

"""

blogwriting_template = f"""

{generate_header(task="Blog Writing")}

Here's an example of a great prompt:

In the context of blog writing, create a comprehensive article that delves into the role of home automation in modern Canadian households.

Cover various facets like intelligent devices, energy conservation, convenience, and heightened security.

Incorporate tangible examples and potential future advancements in this domain.

The article should be well-researched, spanning 800-1,000 words.

Example:

In an age marked by technological progress, home automation stands out as a...

-----

{footer}
"""

twitterpost_template = f"""

{generate_header(task="Writing Twitter Post")}

Here's an example of a great prompt:

As a seasoned content creator, compose an engaging Twitter post that revolves around the themes of AI and automation.
The post should succinctly discuss the latest advancements in AI, particularly focusing on how automation is reshaping industries.

Incorporate relevant hashtags and a concise yet impactful narrative to grab the readers' attention while providing valuable insights.

The output should be a well-crafted Twitter post, not exceeding 280 characters.

Example:

"Exploring the limitless possibilities of AI and automation in today's industries! From autonomous vehicles to smart homes, the future is here. 🤖🏠🚗 #AI #Automation #TechTrends"


-----

{footer}
"""

youtubescript_template = f"""

{generate_header(task="Writing YouTube Scripts")}

Here's an example of a great prompt:

As a master YouTube content creator, develop an engaging script that revolves around the theme of "Exploring Ancient Ruins."

Your script should encompass exciting discoveries, historical insights, and a sense of adventure.

Include a mix of on-screen narration, engaging visuals, and possibly interactions with co-hosts or experts.

The script should ideally result in a video of around 10-15 minutes, providing viewers with a captivating journey through the secrets of the past.

Example:

"Welcome back, fellow history enthusiasts, to our channel! Today, we embark on a thrilling expedition..."

-----

{footer}
"""

PROMPTS = {
    "generate_plan_prompt": PromptTemplate(template=generate_plan_template, input_variables=["text"]),
    "Plan Generator": ChatPromptTemplate.from_template(template=generate_plan_template),
    "Story Telling": ChatPromptTemplate.from_template(template=storywriting_template),  
    "Blog Writing": ChatPromptTemplate.from_template(template=blogwriting_template),
    "Twitter Post": ChatPromptTemplate.from_template(template=twitterpost_template),
    "YouTube Script": ChatPromptTemplate.from_template(template=youtubescript_template)            
}