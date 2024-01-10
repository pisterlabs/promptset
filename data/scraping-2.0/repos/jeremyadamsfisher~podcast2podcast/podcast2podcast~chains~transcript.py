from langchain import LLMChain, OpenAI, PromptTemplate

TAGLINE = "That's all for today. Join us next time for another exciting summary."

FIRST_LINE = """Welcome back. I'm JeremyBot, an artificial intelligence that summarizes \
podcasts. Today we are summarizing {podcast_name}: {episode_name}"""

PROMPT_TEMPLATE_STR = """\
Please write dialog for a talk show where the host, "JeremyBot," discusses and \
summarizes a podcast. There are no guests on this show. Make sure to end with the \
tagline: "{tagline}"

For example, consider the following summary.

Summary: This week on the Slate Gabfest, David Plotz, Emily Bazelon, and John \
Dickerson discuss the House GOP's "Weaponization of Government" subcommittee, \
the insurrection in Brazil, Prince Harry's book "Spare", and the status of \
"return to office". They also provide references and chatters from John, Emily, \
David, and a listener.

Dialog: Welcome! Today we are summarizing The Slate Political Gabfest. On \
this episode, David Plotz, Emily Bazelon, and John Dickerson discuss the House \
GOP's 'Weaponization of Government' subcommittee, the insurrection in Brazil, \
Prince Harry's book 'Spare', and the status of 'return to office'. They also \
provide references and chatters from John, Emily, David, and a listener. \
{tagline}

Summary: {summary}

Dialog: """

PROMPT_TEMPLATE_STR += FIRST_LINE

transcript_template = PromptTemplate(
    input_variables=["summary", "tagline", "podcast_name", "episode_name"],
    template=PROMPT_TEMPLATE_STR,
).partial(tagline=TAGLINE)

transcript_chain = LLMChain(
    llm=OpenAI(temperature=0.0, model_kwargs={"stop": [TAGLINE]}),
    prompt=transcript_template,
)


def generate_transcript(podcast_name: str, episode_name: str, summary: str) -> str:
    """Generate dialog from a podcast summary.

    Args:
        summary (str): The podcast summary.

    Returns:
        str: The dialog.

    """

    # ensure punctuation mark between the end of the prompt and the beginning
    # of the model prediction
    if not any(episode_name.endswith(punc) for punc in ".!?"):
        episode_name = episode_name + "."

    episode_info = {"episode_name": episode_name, "podcast_name": podcast_name}
    llm_output = transcript_chain.predict(summary=summary, **episode_info).strip()
    full_output = FIRST_LINE.format(**episode_info) + " " + llm_output + " " + TAGLINE

    return full_output
