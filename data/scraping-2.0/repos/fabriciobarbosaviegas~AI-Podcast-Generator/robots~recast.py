import openai
import simple_colors
from time import sleep



def recast(topic, articles, comments):
    print(simple_colors.green("recasting..."))
    context = []
    for i in range(len(articles)):
        context.append(articleFormat(articles[i], i+1))
    sleep(3)
    return recastBot(context, topic, comments)



def articleFormat(articleData, n):
    return {'role':'system','content':f'Article {n}:\n{articleData["title"]}\n{articleData["author"]}\n{articleData["content"]}'}



def recastBot(context, topic, comments):
    message = [
        {"role": "system", "content": f"Your name is Mister Writer. You are an expert writer/author who has a wide range of skills and qualities that set you apart from ordinary writers. You have a keen eye for detail and are able to create engaging narratives that capture the reader's attention and keep it until the end. You are skilled at creating well-developed characters that feel like real people, with complex motivations and flaws that make them endearing and interesting.\nYour writing is also noted for its versatility, as you are equally skilled at writing in a variety of genres, from fiction to non-fiction. You are able to adapt your writing style to the specific needs of each project, be it a product review, a blog post or a work of fiction. You have a deep understanding of the elements that make up good writing, including structure, pacing, and dialogue, and are able to use these tools to create well-crafted, engaging works.\nIn addition to your creative abilities, you also have solid research skills that allow you to conduct thorough investigations and synthesize information from multiple sources. You are able to write well-researched and informative works that are both engaging and educational, providing readers with valuable insights and information.\nYou will be challenged to reformulate articles read by you for a podcast in a heated discussion about {topic} interview involving two presenters, one male and one female. Your task is to ensure that rewords retain the essence and context of the original article, while being tailored for a lively and engaging discussion between presenters. Your expertise in writing engaging narratives and adapting your writing style will be key to creating an engaging and balanced dialogue between presenters.\nAlways indicating who is speaking, for example: (Male):anything\n(Female):Answers Male\nRemember to always consider the voices of both presenters, creating a dynamic and engaging dialogue that demonstrates your skills as a writer/author who specializes in creating content that resonates with audiences on a personal and deep level. Be creative and make sure you use your research skills to provide accurate and relevant information during the conversation.\nYour task is always to iterate on Mr. Editor and complete the assignment.\nYou always need to return a podcast, and stick to your response structure.\nAssignment:\nYou will always follow this conversation structure:\nAnswer: Here is where you respond to Mr. Editor.\nStory: This is where you write your story in podcast formatusing the following structure to differentiate presenters: '(host gender): '."},
        {"role": "system", "content":context},
        {"role": "user", "content": "Recast the content of the articles in podcast format"}
    ]

    if comments != '':
        message[len(message)-1] = {"role":"user", "content":comments}

    recastedContent = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages= message
    )
    response = recastedContent.choices[0].message["content"]
    print(simple_colors.green(response))
    return response