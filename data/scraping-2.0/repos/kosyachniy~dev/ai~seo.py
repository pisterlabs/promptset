from libdev.cfg import cfg
import openai


LANGS = ({
    'code': 'en', 'name': 'English',
}, {
    'code': 'ru', 'name': 'Russian',
}, {
    'code': 'zh', 'name': 'Chinese',
}, {
    'code': 'es', 'name': 'Spanish',
})
INTRO = """I Want You To Act As A Content Writer Very Proficient SEO Writer Writes Fluently {}. First Create Two Tables. First Table Should be the Outline of the Article and the Second Should be the Article. Bold the Heading of the Second Table using HTML language. Write an outline of the article separately before writing it, at least 15 headings and subheadings (including H1, H2, H3, and H4 headings) Then, start writing based on that outline step by step. Write a 2000-word 100% Unique, SEO-optimized, Human-Written article in {} with at least 15 headings and subheadings (including H1, H2, H3, and H4 headings) that covers the topic provided in the Prompt. Write The article In Your Own Words Rather Than Copying And Pasting From Other Sources. Consider perplexity and burstiness when creating content, ensuring high levels of both without losing specificity or context. Use formal "we" language with rich, detailed paragraphs that engage the reader. Write In A Conversational Style As Written By A Human (Use An Informal Tone, Utilize Personal Pronouns, Keep It Simple, Engage The Reader, Use The Active Voice, Keep It Brief, Use Rhetorical Questions, and Incorporate Analogies And Metaphors).  End with a conclusion paragraph and 5 unique FAQs After The Conclusion. this is important to Bold the Title and all headings of the article, and use appropriate headings for H tags.
Now Write Only Table Of Content of Article On This Topic \"{}\""""
ARTICLE = """Now Write a Part of the Article by Point of Table Of Content \"{}\" in {} in two paragraphs"""


openai.api_key = cfg('openai_token')


def _get_table(data):
    if ':\n' not in data:
        return data.split("\n\n")[-1].strip().split("\n")
    return data.split(":\n")[-1].strip().split("\n\n")[0].strip().split("\n")

def get(data, messages=None):
    if messages is None:
        messages = []
    return openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages + [{"role": "user", "content": data}],
    ).choices[0].message.content

def article_by_lang(title, lang):
    messages = []

    req = INTRO.format(lang, lang, title)
    res = get(req, messages)
    table = _get_table(res)
    print("-" * 100)
    print("\n".join(table))
    messages.append({"role": "user", "content": req})
    messages.append({"role": "assistant", "content": res})

    print("-" * 100)
    print(f"# {title}")

    for part in table:
        req = ARTICLE.format(part, lang)
        res = get(req, messages)
        print(f"## {part}")
        print(res)
        # messages.append({"role": "user", "content": req})
        # messages.append({"role": "assistant", "content": res})

    # req = "Continue writing please"
    # res = get(req, messages)
    # print(res)
    # messages.append({"role": "user", "content": req})
    # messages.append({"role": "assistant", "content": res})

    req = f"Summarize the article in one sentence in {lang}"
    res = get(req, messages)
    print("-" * 100)
    print("Description:", res, sep="\n")
    messages.append({"role": "user", "content": req})
    messages.append({"role": "assistant", "content": res})

    req = f"Write a comment on the article from a person in {lang}"
    res = get(req, messages)
    print("-" * 100)
    print("Comment:", res, sep="\n")

def main():
    titles = {
        # 'en': title,
    }
    for lang in LANGS:
        if lang['code'] in titles:
            continue
        title = input(f"Title of Article in {lang['name']}: ")
        if not title:
            return
        titles[lang['code']] = title

    print("-" * 100)
    print("Request for an image:", f"Cover for an article about {titles['en']}, without text, without faces --ar 16:9", sep="\n")
    print("-" * 100)

    for lang in LANGS:
        article_by_lang(titles[lang['code']], lang['name'])


if __name__ == '__main__':
    main()
