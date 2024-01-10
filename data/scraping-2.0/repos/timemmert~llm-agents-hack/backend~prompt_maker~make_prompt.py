from langchain import OpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate


# first initialize the large language model
llm = OpenAI(
    temperature=0,
    model_name="text-davinci-003",
    openai_api_key="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",
)

# now initialize the conversation chain

prompt_template = PromptTemplate(
    input_variables=["history", "input"],
    template="Given a detailed description of a \
    person, please write a bulleted \
    point summary about \
    1. Their personality\
    2. Their values \
    3. Their background (educational, where they grew up, etc.) \
    4. Their age \
    5. Their name \
    6. Their hobbies \
    7. Their goals \
    8. Their political leanings. \
    Format this like    \
    1. Personality\
    2. Values \
    3. ...\
    Do not hallucinate facts. If you do not know something about the user (e.g., \
    their age), make an educated guess but don't hallucinate. \
    Current conversation: {history} \
    Human: {input} \
    AI:",
)
conversation_buf = ConversationChain(
    llm=llm, memory=ConversationBufferMemory(), prompt=prompt_template
)
print(
    conversation_buf(
        # "Hi! My name is Nikhil, and I'm a BS/MS student at Stanford. "
        #         """
        # Hey there! I'm Sarah, and I'm a 16-year-old from a small town in Indiana. I come from a big, close-knit family with three brothers and two sisters, so things can get pretty chaotic at home, but I love it.
        # I'm really into sports, and I play on my high school's soccer team. I've been playing soccer since I was a kid, and I even scored the winning goal in the championship game last year. It was such an epic moment!
        # When I'm not on the soccer field, you can usually find me hanging out at our local library. I'm a total bookworm, and I love getting lost in different worlds through novels. My all-time favorite series is "Harry Potter" - I've read it like a hundred times!
        # I'm also a bit of a foodie. I enjoy experimenting with cooking and trying out new recipes. My specialty is homemade chocolate chip cookies - they're seriously the best.
        # Oh, and I love music too. I play the guitar and enjoy strumming some tunes in my free time.
        # So, that's a bit about me, just a regular teenager from Indiana, trying to balance sports, books, and baking, and having a blast doing it!
        # """
        #         """
        # I am Barack Obama, and as the 44th President of the United States, I had the incredible privilege of serving this nation from 2009 to 2017. My journey to the White House was a testament to the enduring promise of the American dream.
        # I was born on August 4, 1961, in Honolulu, Hawaii. My diverse background – a Kenyan father and a Kansan mother – has shaped my perspective and informed my commitment to inclusivity and unity. I often reflect on the values instilled in me by my parents and grandparents, emphasizing hard work, education, and community involvement.
        # My time as President was marked by several significant achievements. The passage of the Affordable Care Act, also known as Obamacare, expanded access to healthcare for millions of Americans. We took significant steps in addressing climate change through the Paris Agreement and encouraged clean energy innovation. I also sought to strengthen relationships with our international allies and promote diplomacy and multilateral cooperation.
        # Throughout my tenure, I aimed to bridge divisions and work toward a more inclusive and equitable America. My administration saw the end of the "Don't Ask, Don't Tell" policy, allowing LGBTQ+ individuals to serve openly in the military. We also made strides in criminal justice reform, with the signing of the First Step Act.
        # The 2008 financial crisis was a significant challenge during my presidency, and my administration worked to stabilize the economy and prevent a further downturn. I aimed to provide economic opportunities for all, believing in the importance of lifting people out of poverty and expanding the middle class.
        # It was an honor and a privilege to serve as President, and my hope was to inspire young people to engage in civic life, emphasizing the power of change through community involvement and political participation. The road was not always easy, and there were many challenges to overcome, but I remained committed to the belief that, in America, we could achieve greatness through unity, empathy, and hard work.
        # I continue to be involved in public life, advocating for the values and ideals I hold dear. My time in office may be over, but my commitment to making a positive impact on the world remains steadfast.
        # """
        """
Well, howdy! I'm Mike, a 42-year-old gearhead from Wasilla, Alaska. I've been living up here for most of my life, and I wouldn't trade the Alaskan wilderness for anything. 

I work as a mechanic at the local auto shop, and I've got a real knack for fixing things. Cars, trucks, snowmobiles - you name it, and I can probably get it running like a dream. I guess you could say I've got grease in my veins.

When I'm not wrenching on engines, you'll find me out in the great outdoors. I'm an avid fisherman, and there's nothing like spending a day on the river or out on the lake, trying to reel in the big one. I've got a freezer full of salmon, and I'm always up for a good fishing story exchange.

I also like to take my snowmobile out in the winter - it's the best way to get around when the snow piles up. Sometimes, I'll even take my kids out for a little winter adventure. I've got two of 'em, a 14-year-old daughter and an 11-year-old son. They're already showing some interest in the mechanic life, and I couldn't be prouder.

Being from Wasilla, you can't help but love the rugged Alaskan lifestyle. It's a unique place to call home, and I wouldn't have it any other way. So, that's me in a nutshell, just a mechanic from the Last Frontier, making sure folks can get where they need to go, and enjoying the wild beauty of Alaska in my spare time.
"""
    )
)
