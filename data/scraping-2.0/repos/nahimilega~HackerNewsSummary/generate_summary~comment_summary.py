import openai


context = """You are an expert summarizer working for a Hacker News."""

prompt = """Here is an article posted on Hacker News with all the Hacker News comments
### Title: {title}

### Hacker News Comments:
{comments}


Please summarize discussion in comments.


Bullet point the main topics and key points discussed. Capture the range of opinions expressed in the comments, reflecting the depth and variety of discussions that occur on Hacker News. 

Try to quote the comments and rebuttle of those comments. Make it more like gossip and spicy

Avoiding Boilerplate Language. Do not mention article name or start with "The comments on the article..."

Do now repeat the title of the article in the summary.

Remove any fluff and keep only dense to the point information.

Keep it short.

REMEMBER: Keep it in pointers and very concise.

"""


def make_coment_summary_prompt(story_data) -> str:
    comment_list = "\n".join(story_data.comment_list)
    comment_list = comment_list.replace("\n", "\n> ")
    comment_list = comment_list[:20000]
    return prompt.format(
        title=story_data.title,
        url=story_data.url,
        comments=comment_list,
    )


def make_summary(story_data, retry_count: int = 0) -> str:
    userPrompt = make_coment_summary_prompt(story_data)
    messages = [
        {"role": "system", "content": context},
        {"role": "user", "content": userPrompt},
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-1106",
        messages=messages,
        temperature=0 + retry_count * 0.1,
        max_tokens=1000,
        presence_penalty=0,
        frequency_penalty=0,
        # auto is default, but we'll be explicit
    )
    return response["choices"][0]["message"]["content"]


def generate_comment_summary(story_data) -> str:
    retry_count = 0
    while retry_count < 4:
        retry_count += 1
        try:
            return make_summary(story_data, retry_count)
        except Exception as e:
            print(f"Failed to generate comment summary for {story_data.url}: {e}")

    return ""
