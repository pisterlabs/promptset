import openai
from django.conf import settings

openai.api_key = settings.OPENAI_API_KEY


def generate_blog_to_topic_ideas(audience, topic, keywords):
    blog_topics = []

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="Generate 8 Blog topic ideas on the following topic: {}\naudience \nkeywords {} \n*".format(audience,
                                                                                                           topic,
                                                                                                           keywords),
        temperature=0.8,
        max_tokens=300,
        top_p=1,
        best_of=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    if "choices" in response:
        if len(response["choices"]) > 0:
            res = response["choices"][0]["text"]
        else:
            res = []
    else:
        res = []
    a_list = res.split("*")
    if len(a_list) > 0:
        for blog in a_list:
            blog_topics.append(blog)
    else:
        res = []
    return blog_topics


def generate_blog_to_section_titles(audience, topic, keywords):
    blog_section = []

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="Generate Adequate blog section titles for the provided blog topic audience and keywords: {}\naudience \n{}keywords {}\n*".format(
            audience, topic, keywords),
        temperature=0.8,
        max_tokens=300,
        top_p=1,
        best_of=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    if "choices" in response:
        if len(response["choices"]) > 0:
            res = response["choices"][0]["text"]
        else:
            res = []
    else:
        res = []
    a_list = res.split("*")
    if len(a_list) > 0:
        for blog in a_list:
            blog_section.append(blog)
    else:
        res = []
    return blog_section


def generate_blog_section_detail(blogTopic, sectionTopic, audience, keywords, profile):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="Generate detailed blog section write up for the following blog section keading, using the blog title, audience and keywords provided.{}\nBlog Section Heading: {}\nBlog Title {}\nAudience: {}\nkeywords: \n".format(
            blogTopic, sectionTopic, audience, keywords
        ),
        temperature=0.8,
        max_tokens=500,
        top_p=1,
        best_of=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    if "choices" in response:
        if len(response["choices"]) > 0:
            res = response["choices"][0]["text"]
            if not res == "":
                cleanedRes = res.replace("\n", "<br>")
                if profile.monthlyCount:
                    oldCount = int(profile.monthlyCount)
                else:
                    oldCount = 0
                oldCount += len(cleanedRes.split(" "))
                profile.monthlyCount = str(oldCount)
                profile.save()
                return cleanedRes
            else:
                return ""
        else:
            return ""
    else:
        return ""


def check_count_allowance(profile):
    if profile.subscribed:
        type = profile.subscriptionType
        if type == "free":
            max_limit = 5000
            if profile.monthlyCount:
                if int(profile.monthlyCount) < max_limit:
                    return True
                else:
                    return False
            else:
                return True
        elif type == "starter":
            max_limit = 40000
            if profile.monthlyCount:
                if int(profile.monthlyCount) < max_limit:
                    return True
                else:
                    return False
            else:
                return True
        elif type == "advanced":
            return True
        else:
            return False
    else:
        max_limit = 5000
        if profile.monthlyCount:
            if int(profile.monthlyCount) < max_limit:
                return True
            else:
                return False
        else:
            return True
