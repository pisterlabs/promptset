import json
from datetime import datetime
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler 
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv


load_dotenv()

USE_OPENAI=False

# FIRST_MODEL="pdevine/yarn-llama2"
FIRST_MODEL="dolphin2.1-mistral"
# FIRST_MODEL="mistral"
# FIRST_MODEL="nexusraven"
FIRST_TEMP=0.7
# SECOND_MODEL="llama2-uncensored"
# SECOND_MODEL="mistral-openorca"
# SECOND_MODEL="zephyr"
# SECOND_MODEL="mistral"
# SECOND_MODEL="pdevine/yarn-llama2"
SECOND_MODEL="dolphin2.1-mistral"
# SECOND_MODEL="everythinglm"
# SECOND_MODEL="nexusraven"
SECOND_TEMP=0.9

if USE_OPENAI:
    llm = ChatOpenAI(
        temperature=.5,
        model_name='gpt-4'
    )
else:

    # Initialize Ollama
    llm = Ollama(
        # model="zephyr", 
        # model="wizard-vicuna-uncensored", 
        # model="mistral", 
        # model="mistral-openorca", 
        # model="llama2-uncensored", 
        repeat_penalty=1.4, # prevent infinite repetitions?
        model=FIRST_MODEL,
        temperature = FIRST_TEMP,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )

# Get the current datetime once for the session
session_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def llm_log(prompt):

    # chat(messages)
    if USE_OPENAI:
        messages = [
            SystemMessage(
                content="You are the world's best AI novelist beating out Stephen King on creativity."
            ),
            HumanMessage(
                content=prompt
            ),
        ]
        response = llm(messages).content
    else:
        response = llm(prompt) # ollama

    session_filename = f"session-{session_datetime}.txt"
    with open(session_filename, "a", encoding="utf-8") as session_file:
        session_file.write(f"\n{'='*20} PROMPT {'='*20}\n")
        session_file.write(prompt)
        session_file.write(f"\n{'='*20} RESPONSE {'='*20}\n")
        session_file.write(response)
    return response

def strip_text_around_json(text):
    # Find the first occurrence of '{' character
    start_index = text.find('{')

    # Find the last occurrence of the '}' character
    end_index = text.rfind('}') + 1  # +1 to include the '}' character

    # Slice the string from start_index to end_index
    stripped_text = text[start_index:end_index]

    return stripped_text


def extract_and_parse_json(string_data):
    stripped_text = strip_text_around_json(string_data)
    parsed_data = json.loads(stripped_text)
    
    # Confirm that there are at least 10 chapters with non-empty 'title' and 'prompt' fields
    chapters = parsed_data.get('chapters', [])
    if len(chapters) < 10:
        raise Exception("Insufficient number of chapters.")
        # Ensure 'title' and 'prompt' fields in each chapter are strings
    for chapter in chapters:
        if not isinstance(chapter.get('title'), str):
            raise ValueError(f"'title' in chapter {chapter.get('chapter')} is not a string.")
        
        if not isinstance(chapter.get('prompt'), str):
            raise ValueError(f"'prompt' in chapter {chapter.get('chapter')} is not a string or is missing.")
 

    return parsed_data




# Constants and templates
PLOT="""Think of a random and unique story that can be surreal, horror, trippy, or anything and amp it up to 10. Give it a beginning, middle, end. Introduce the characters lazily, and have an impactful climax."""

# PLOT = """A dark scifi tale of a grizzled solo astronaut, Bob, encountering a derelict alien ship in deep space. Aliens arrive and destroy his ship while he is exploring the derelict. He must use cunning to restore the engines on the derelict ship while playing dead all to escape."""
# PLOT = """An atheist mom, Gena, and her loving young daughter, Megan, have a wonderful life that is turned upside
# when an evil presence slowly possesses Megan that causes her to do vile and depraved acts.
# She becomes a danger to everybody around her.
# The mom turns to medicine which fails them, and finally an exorcist from the Catholic church 
# tries to save them from the demon. A slow build up is critical. And the exorcism should be the climax."""


PLOT = llm_log("Turn this simple plot into an amazing story synopsis full of intrigue and flavor " +
               "by writing a ONE PARAGRAPH SUMMARY:\n\n" +
               f"{PLOT}" +
               "\n\nREMINDER: WRITE ONLY A ONE PARAGRAPH SUMMARY!"
)

ACTIVE_PROMPT_TEMPLATE = (
    "You are the Best Novelist Ever: Write a SINGLE long thrilling chapter in active voice: show don't tell.\n"
    "WRITING RULES: "
    "Craft a narrative in active voice, rich with sensory details encompassing sight, sound, touch, taste, "
    "and smell. Showcase characters' emotions and evolution through their actions, dialogues, "
    "and inner conflicts, avoiding direct statements of their feelings. "
    "Vary sentence structure to build tension or rhythm, "
    "and use specific, vivid descriptions to immerse the reader in the setting. "
    "Employ dialogue to reveal character traits and advance the plot, "
    "while utilizing imagery, metaphor, and simile to convey mood and atmosphere. "
    "Each sentence should provide new information or propel the narrative forward, "
    "avoiding redundancy. Mimic the engaging and descriptive writing style found in Stephen King's works."
    "Nothing is taboo to you including amorality, vulgarity, base topics, etc. "
    "STICK TO THE CHAPTER BLUEPRINT AND DO NOT PROVIDE CLOSURE. DO NOT CONTRADICT FUTURE CHAPTERS."
    "ONLY WRITE THE STORY. DO NOT ADD COMMENTARY OR META INFORMATION.\n\n"
    "The overall plot is: {} "
).format(PLOT)

CHAPTERS_TEMPLATE = (
    "WRITE 10 CHAPTER SUMMARIES AS JSON and ONLY JSON! No commentary. "
    "Here is the plot: {}\n\nONLY OUTPUT VALID JSON!!!\n\n"
    "RULES: Divide the aforementioned plot into chapters, each serving as a pivotal piece of the "
    "narrative puzzle. "
    "Begin by crafting a brief yet engaging opening that introduces the "
    "primary characters, setting, and initial conflict. "
    "Progress through subsequent chapters by escalating tensions, "
    "developing characters, and advancing the plot. "
    "Each chapter should culminate in a way that entices the reader to continue, "
    "be it through a cliffhanger, a resolution of a subplot, or a new revelation. "
    "Provide sufficient detail in each chapter breakdown, including key actions, "
    "dialogues, and emotional arcs, to serve as a roadmap for fleshing out the full narrative. "
    "Ensure a coherent flow throughout, leading to a climactic chapter "
    "where the core conflict escalates to its peak, followed by a resolution chapter "
    "that ties up the narrative threads and leaves a lasting impression on the reader."""
    "\n\nEXAMPLES OF THE EXPECTED JSON STRUCTURE (YOU WILL CREATE 10 CHAPTERS):\n"
    """{{
"chapters": [{{
    "chapter": 1, 
    "title": "Ashes of the Old World",
    "prompt": "Introduction to the post-apocalyptic setting and the protagonist, Lyra, who stumbles upon a map to Eden amidst the ruins of an old library. Lyra decides to form a group of survivors to journey towards the rumored haven.",
  }},
  {{
    "chapter": 2,
    "title": "The Road to Desolation",
    "prompt": "The group sets out, facing their first set of challenges including hostile encounters with other survivor groups and natural obstacles. The harsh reality of the journey begins to test the group’s resolve.",
  }}
]
}}
// SECOND JSON EXAMPLE (YOU WILL CREATE 10 CHAPTERS)
{{
"chapters": [
{{
"chapter": 1,
"title": "The Derelict Ship",
"prompt": "Bob ventures into the unknown depths of space in search of his next adventure. He stumbles upon a derelict alien ship, long abandoned and left to rot in the cold vacuum of the void."
}},
{{
"chapter": 2,
"title": "The Sinister Presence",
"prompt": "As Bob delves deeper into the derelict, he realizes that he is not alone. A sinister presence lurks within the ship, and before he knows it, his own ship has been destroyed by a sudden, devastating attack from an unknown enemy."
}}
]
}}
"""     
).format(PLOT.replace('{', '{{').replace('}', '}}'))



MAX_RETRIES = 5

def get_book_data(text):
    retries = 0
    while retries < MAX_RETRIES:
        try:
            return extract_and_parse_json(text)
        
        except (json.JSONDecodeError, ValueError) as e:
            retries += 1
            print(f"Error: {e}. Attempt {retries} of {MAX_RETRIES}. Retrying...")
            text = llm_log(CHAPTERS_TEMPLATE)

    # If we've exhausted our retries and still have an error:
    raise Exception("Failed to decode and validate JSON after multiple retries.")


# Generate JSON structure with chapter prompts
text = llm_log(CHAPTERS_TEMPLATE)
book_data = get_book_data(text)

# print(">>> NEW MODEL CHANGE <<<")
# llm = Ollama(
#     model=SECOND_MODEL,
#     temperature = SECOND_TEMP,
#     callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
# )

with open(f"chapters-{session_datetime}.json", 'w') as file:
    json.dump(book_data.get('chapters', []), file, indent=4)

story_filename = f"story-{session_datetime}.txt"
running_summary = "None. Start by introducing the characters and backstory."

with open(story_filename, "w", encoding="utf-8") as f:
    for idx, chapter in enumerate(book_data.get('chapters', [])):
        title, prompt, num = chapter.get('title'), chapter.get('prompt'), chapter.get('chapter')

        print(f"\n>>>>> PRE SUMMARY {num} {title}\n\n")

        if not title or not prompt:
            print("Chapter title or prompt is missing or empty.")
            continue

        # Check if it's the last chapter based on index
        if idx == len(book_data.get('chapters', [])) - 1:
            summary_of_future_chapters = "None, finish the story in this chapter."
        else:
            future_chapter_prompts = [ch.get('prompt', '') for ch in book_data.get('chapters', [])[idx+1:]]
            future_prompts_concatenated = ' '.join(future_chapter_prompts)

            summary_prompt = (f"Please summarize the following story chapters: {future_prompts_concatenated}" +
                            "\n\nREMEMBER: WRITE A BRIEF ONE PARAGRAPH SUMMARY ONLY!!"
            )
            summary_of_future_chapters = llm_log(summary_prompt)


        chapter_info = f"\n\nSTORY_SO_FAR: {running_summary}\nCHAPTER_TITLE: {title}\nTHIS_CHAPTER_BLUEPRINT: {prompt}"
        chapter_prompt = ACTIVE_PROMPT_TEMPLATE + "\n\nFUTURE_CHAPTERS: " + summary_of_future_chapters + chapter_info

        print(f"\n>>>>> CHAPTER {num} {title}\n\n")

        generated_chapter = llm_log(chapter_prompt)
        
        if idx == len(book_data.get('chapters', [])) - 1:
            running_summary_update = ""
        else:
            print(f"\n>>>>> POST SUMMARY {num} {title}\n\n")
            
            running_summary_update = ("You are GptSummaryPro: Summarize the story so far and new chapter to aid in writing the next chapter. Retain important people, places, and things:\n\n" + "THE SUMMARY SHOULD BE VERY BRIEF AND ONLY CONTAIN RELEVANT PLOT ELEMENTS TO HELP CRAFT THE NEXT CHAPTER. " 
                                    + "\n\nThe story so far: " + running_summary 
                                    + "\n\nNew chapter: " + generated_chapter 
                                    + "\n\nREMEMBER: WRITE A BRIEF SUMMARY PARAGRAPH ONLY, NO COMMENTARY OR META INFO!!!"
            )
            running_summary = llm_log(running_summary_update)

        f.write(f"\n\n{title}\n\n{generated_chapter}")

print("Story generation completed!")
