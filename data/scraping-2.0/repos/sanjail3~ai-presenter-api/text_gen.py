import os
import openai
from dotenv import load_dotenv




# Generate the response using OpenAI API
def get_response(topic):
    load_dotenv()
    openai_secret_key=os.getenv("OPEN_API_KEY")




    openai.api_key = openai_secret_key
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"""
    I am making slide presentation on {topic}  give me content of 7 slides  each slide should have concepts explainaton in a  detailed manner for  topic title of each slide in points  your response should follow this pattern  for any topics 

    Slide 1: Title Slide
    Title: 

    Slide 2: Topics
    Topics to be discussed:


    Slide 3:
    Topic: 
    Content:


    Slide 4:
    Topic: 
    Content:


    Slide 5:
    Topic: 
    Content:


    Slide 6:
    Topic:
    Content:


    Slide 7:
    Topic: 
    Content:



    Slide 8:
    Topic: 
    Content:


    Slide 9:
    Topic: 
    Content:

    """

             }
        ]
    )

    # Extract the text for each slide
    slides = response.choices[0].message.content.split("Slide")
    print(slides)

    formatted_slides = []
    for slide in slides[4:]:
        # Split slide into title and content
        slide_parts = slide.split("Content:")
        if len(slide_parts) > 1:
            val = slide_parts[0].split("Topic:")
            title = val[1]
            content = slide_parts[1].strip()
            formatted_slide = {
                "title": title,
                "content": content
            }
            formatted_slides.append(formatted_slide)

    # Print the formatted slides
    slide_t = []
    slide_c = []
    for index, slide in enumerate(formatted_slides):
        slide_t.append(slide['title'])
        slide_c.append(slide['content'])

    slide1_title = slides[2].split("Title:")[1]
    slide2_content = slides[3].split("Topics to be discussed:")[1]

    return  slide1_title,slide2_content,slide_t,slide_c

