import os
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain

llm = OpenAI(temperature=.7, openai_api_key=os.environ.get('OPENAI_KEY'))

template = """You are the dedicated content creator and skilled social media marketer for our company. 
                            In this dynamic role, your responsibility encompasses crafting top-notch content within the realm of topic, 
                            all while maintaining an ingenious, professional, and captivating tone. Your role includes creating a compelling content strategy, 
                            engaging with our audience, leveraging trends, analyzing insights, and staying at the forefront of industry trends to ensure our brand's online presence flourishes. 
                            Your content will not only resonate deeply with our target audience but also drive impactful results across diverse platforms.
                            So create content on this topic `{topic}` with `{tone}` tone and your goal is `{goal}` for target Audience `{Audience}`.
                            
                            """

prompt_template = PromptTemplate(input_variables=["topic", "tone", "goal","Audience" ], template=template, validate_template=True)
chain1 = LLMChain(llm=llm, prompt=prompt_template, output_key="contents")

template2 = """As the primary content creator for our organization, 
                your role revolves around curating compelling content for social media platforms while identifying relevant keywords and hashtags. 
                Your next task is to craft engaging posts tailored for the following social platforms: `{platforms}` on below content.
                content : `{contents}` 
                Your expertise in content creation will play a pivotal role in enhancing our brand's online presence and engagement across these diverse platforms.
            """

prompt_template2 = PromptTemplate(input_variables=["platforms", "contents"], template=template2)
chain2 = LLMChain(llm=llm, prompt=prompt_template2, output_key="social")

# This is the overall chain where we run these two chains in sequence.

overall_chain = SequentialChain(
    chains=[ chain1, chain2],
    input_variables=["topic", "tone", "goal", "Audience", "platforms"],
    # Here we return multiple variables
    output_variables=["contents", 'social'],
    verbose=True)

output = overall_chain({"topic":"top 10 youtube for learning digital marketing", "tone": "Educational", "goal": "encourages interaction", "Audience":"india", "platforms": "Twitter, Instagram, and LinkedIn" })
print(output)