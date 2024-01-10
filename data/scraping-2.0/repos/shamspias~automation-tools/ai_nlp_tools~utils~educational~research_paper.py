from django.conf import settings
import openai

openai.api_key = settings.OPEN_AI_KEY


def generate_research_paper_topics(prompt):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt="Generate research paper topics on: {}. \n\n 1.".format(prompt),
        temperature=0.7,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    paper_title_list = response['choices'][0]['text'].split("\n")
    context = {
        'data': [word_value[3:] if word_value[0] != " " else word_value[4:] if i != 0 else word_value[1:] for
                 i, word_value in enumerate(paper_title_list) if word_value != ""]
    }
    return context


def research_paper_sections(prmt):
    """
    :param prmt:
    :return: context dist
    """
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt="Expand the research paper title into high-level sections where the title: {}.\n".format(prmt),
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    my_text = response['choices'][0]['text'].split("\n")
    context = {
        'data': [value[2:] for value in my_text]
    }
    return context


def research_paper_section_expander(section, title):
    """
    :param title:
    :param section:
    :return: context dist
    """
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt="Expand the research paper {} section into a detailed professional, witty and clever explanation where "
               "the title: {}.\n".format(section, title),
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    my_text = response['choices'][0]['text'].split("\n\n")
    context = {
        'data': [i for i in my_text if not (i == "" or i == " ")]
    }
    return context
