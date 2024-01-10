import openai
import os 

openai.api_key = os.environ.get("OPENAI_KEY")

def openAiGenerate(language, feature, tone, input_message):
    prompt_list = {
        "th" : {
            "เขียนแคปชั่นขายของ" : """Write a social media announcement about [{input}] with hashtags and emojis The feeling of the message should be [{type}]. [เป็นภาษาไทยเท่านั้น]""",
            "ช่วยคิดคอนเทนต์" : """Create list of idea content with short biref about [ {input}] that all content should make feeling like [ {type}] 
        show list of idea with short biref [เป็นภาษาไทยเท่านั้น]:""",
            "เขียนบทความ": """Write a blog post with high demand SED keyword that talks about [main topic of article] that article should feel like [emotion of message] [เป็นภาษาไทย]:
        main topic of article:  {input}
        emotion of message:  {type}""",
            "เขียนสคริปวิดีโอสั้น" : """write full scripts for short video that talk about [ {input}] and the feeling of scripts is [ {type}] [เป็นภาษาไทยเท่านั้น]:""",
            "เขียนประโยคเปิดคลิป" : """Compose a Captivating Clickbait Sentence but not incloud 'Click' in Sentence for Openning a Short Video To Talk About [ {input}] And Look [ {type}] That Instantly Grabs the Viewer's Attention and Sets the Stage for an Unforgettable Experience [เป็นภาษาไทยเท่านั้น]"""
        },
        "eng" : {
            "เขียนแคปชั่นขายของ" : "Write a social media announcement about [ {input}] and the feeling of message is [ {type}]:",
            "ช่วยคิดคอนเทนต์" : """Create list of idea content with short biref about [ {input}] that all content should make feeling like [ {type}] 
        show list of idea with short biref:""",
            "เขียนบทความ": "Write a blog post with high demand SED keyword that talks about [ {input}] that article should feel like [ {type}]:",
            "เขียนสคริปวิดีโอสั้น" : "write full scripts for short video that talk about [ {input}] and the feeling of scripts is [ {type}]:",
            "เขียนประโยคเปิดคลิป" : """Compose a Captivating CClickbait Sentence but not incloud 'Click' in Sentence for Openning a Short Video To Talk About [ {input}] And Look [ {type}] That Instantly Grabs the Viewer's Attention and Sets the Stage for an Unforgettable Experience:"""
        },
        "id" : {
            "เขียนแคปชั่นขายของ" : "Write a social media announcement about [ {input}] and the feeling of message is [ {type}] [in Bahasa Indonesia Only]:",
            "ช่วยคิดคอนเทนต์" : """Create list of idea content with short biref about [ {input}] that all content should make feeling like [ {type}] 
        show list of idea with short biref [in Bahasa Indonesia Only]:""",
            "เขียนบทความ": "Write a blog post with high demand SED keyword that talks about [ {input}] that article should feel like [ {type}] in [Bahasa Indonesia]:" ,
            "เขียนสคริปวิดีโอสั้น" : "write full scripts for short video that talk about [ {input}] and the feeling of scripts is [ {type}] [in Bahasa Indonesia Only]:" ,
            "เขียนประโยคเปิดคลิป" : """Compose a Captivating Clickbait Sentence but not incloud 'Click' in Sentence for Openning a Short Video To Talk About [ {input}] And Look [ {type}] That Instantly Grabs the Viewer's Attention and Sets the Stage for an Unforgettable Experience [in Bahasa Indonesia Only]"""
        }
    }

    prompt = prompt_list[language][feature]

    prompt = prompt.format(
        input = input_message,
        type = tone
    ) 
    
    result = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt},
        ]
    )
    assistant_reply = result['choices'][0]['message']['content']

    return assistant_reply
