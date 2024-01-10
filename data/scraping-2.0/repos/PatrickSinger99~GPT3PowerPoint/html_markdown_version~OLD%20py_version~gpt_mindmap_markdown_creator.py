from gpt_content_creator import create_text_from_topic
from jupyter_notebook import gpt_subtopic_creator
from gpt import GPT
import openai
from gpt import Example

# Creates the html body wrapper around <section> parts
from jupyter_notebook.gpt_subtopic_creator import create_subtopics_from_topic


def pack_sections_in_body(sections_list):
    body_start = "<body class=\"jp-Notebook\"><div class=\"reveal\"><div class=\"slides\">"
    body_end = "</div></div></body>"

    sections = "".join(sections_list)

    return body_start + sections + body_end


# Creates html jupyter notebook presentation code
def create_markdown_presentation(sections_list):
    # Static Head Code (Start)
    with open("markdown_html_header.txt", encoding="utf8") as head:
        header = head.read()

    # GPT generated body code
    with open("markdown_script_part.txt", encoding="utf8") as end:
        ending = end.read()

    # Static Script Code (Ending)
    with open("new_html_presentation.html", "w",  encoding="utf8") as html:
        html.write(header)
        html.write(pack_sections_in_body(sections_list))
        html.write(ending)


# Openai key
with open("openai_key.txt") as file:
    key = file.read()
    openai.api_key = key

# Get topic from user and call create_text_from_topic function which generates bulletpoints from the topic
prompt = input("Topic: ")

content = create_subtopics_from_topic(prompt)

# GPT model to create html code from heading and bulletpoints
gpt_mindmap_markdown_creation = GPT(engine="davinci", temperature=.5, max_tokens=120)

gpt_mindmap_markdown_creation.add_example(Example("internet, computer, communication, network, ethernet, router, backbone",
                                          "<ul><li>internet</li><li>computer</li><li>communication</li><li>network</li><li>ethernet</li><li>router</li><li>backbone</li></ul>"))

gpt_mindmap_markdown_creation.add_example(Example("vehicle, transportation, road, wheels",
                                          "<ul><li>vehicle</li><li>transportation</li><li>road</li><li>wheels</li></ul>"
                                        ))

gpt_mindmap_markdown_creation.add_example(Example("social network, internet, user, privacy",
                                          "<ul><li>social network</li><li>internet</li><li>user</li><li>privacy</li></ul>"
                                        ))

gpt_mindmap_markdown_creation.add_example(Example("technology, hardware, software, internet, microchip, silicon, chip",
                                          "<ul><li>technology</li><li>hardware</li><li>software</li><li>internet</li><li>microchip</li><li>silicon</li><li>chip</li></ul>"
                                        ))


# Create html Code from GPT Model and create presentation
output = gpt_mindmap_markdown_creation.submit_request(content)
section_html = output.choices[0].text[8:]
print("GPT-3 generated html:\n" + section_html)
#create_markdown_presentation([section_html])

