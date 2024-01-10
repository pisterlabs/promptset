from gpt_content_creator import create_text_from_topic
from gpt_subtopic_creator import create_subtopics_from_topic
from gpt import GPT
import openai
from gpt import Example


# Creates the html body wrapper around <section> parts
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

# GPT model to create html code from heading and bulletpoints
gpt_markdown_creation = GPT(engine="davinci", temperature=.5, max_tokens=120)

gpt_markdown_creation.add_example(Example("Heading: Cat Text: The cat (Felis catus) is a domesticated species of small carnivorous mammal. It is the only domesticated species in the family Felidae.",
                                       "<section><div class=\"jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput \" data-mime-type=\"text/markdown\"><h1>Cat</h1><ul><li>The cat (Felis catus) is a domesticated species of small carnivorous mammal.</li><li>It is the only domesticated species in the family Felidae.</li></ul></div></section>"
                                        ))

gpt_markdown_creation.add_example(Example("Heading: PyCharm Text: PyCharm is an integrated development environment used in computer programming. It is developed by the Czech company JetBrains.",
                                          "<section><div class=\"jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput \" data-mime-type=\"text/markdown\"><h1>PyCharm</h1><ul><li>PyCharm is an integrated development environment used in computer programming.</li><li>It is developed by the Czech company JetBrains.</li></ul></div></section>"
                                        ))

gpt_markdown_creation.add_example(Example("Heading: python Text: Python is a high-level programming language. It is a general-purpose language. It is a high-level language. It is a widely used language.",
                                          "<section><div class=\"jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput \" data-mime-type=\"text/markdown\"><h1>python</h1><ul><li>Python is a high-level programming language.</li><li>It is a general-purpose language.</li><li>It is a high-level language.</li><li>It is a widely used language.</li></ul></div></section>"
                                        ))

gpt_markdown_creation.add_example(Example("Heading: Etymology and naming Text: The origin of the English word cat is thought to be the Late Latin word cattus, which was first used at the beginning of the 6th century.",
                                          "<section><div class=\"jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput \" data-mime-type=\"text/markdown\"><h1>Etymology and naming</h1><ul><li>The origin of the English word cat is thought to be the Late Latin word cattus, which was first used at the beginning of the 6th century.</li></ul></div></section>"
                                        ))


# Get topic from user
prompt = input("Topic: ")


# Get subtopics from topic
subtopics = create_subtopics_from_topic(prompt)
subtopics = subtopics.split(", ")

subtopics_list = [prompt]
for i in subtopics:
    subtopics_list.append(i.strip())

if len(subtopics_list) > 5:
    subtopics_list = subtopics_list[:4]

print("GPT-3 generated subtopics:\n", subtopics_list)


# Get text for every subtopic
section_list = []
for topic in subtopics_list:

    if prompt not in topic:
        topic = prompt + " " + topic

    print("\n" + topic)
    print("---------------")

    content = create_text_from_topic(topic)

    # Clean up content output
    content.replace("\"", "")
    content.replace("output:", "")

    # Convert topic and bulletpoints to format for the html GPT model
    content_input = "Heading: " + topic + " Text: " + content

    # Create html Code from GPT Model and create presentation
    output = gpt_markdown_creation.submit_request(content_input)

    # clean up output code
    section_html = output.choices[0].text[8:]
    section_html.replace("output:", "")
    section_html = section_html.strip()
    section_html = section_html.split("</section>")[0] + "</section>"

    print("GPT-3 generated html:\n" + section_html)
    section_list.append(section_html)

create_markdown_presentation(section_list)
