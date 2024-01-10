from openai import OpenAI
import json
from dotenv import load_dotenv
import os
import requests
from pexels_api import API
import random

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = os.getenv("OPENAI_MODEL")


def search_and_download(search_term, filename='image.jpg'):
    PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
    api = API(PEXELS_API_KEY)
    api.search(search_term)
    photos = api.get_entries()

    if photos:
        # Limit to the top 5 results
        top_photos = photos[:5]

        # Select a random image from the top 5 results
        if top_photos:
            selected_photo = random.choice(top_photos)
            image_url = selected_photo.original

            # Download the selected image
            img_data = requests.get(image_url).content
            with open(filename, 'wb') as handler:
                handler.write(img_data)
            return json.dumps({"search_term": search_term, "image_url": image_url, "saved_as": filename})
        else:
            return json.dumps({"search_term": search_term, "image_url": "None", "saved_as": "None"})
    else:
        return json.dumps({"search_term": search_term, "image_url": "None", "saved_as": "None"})


def run_images(keyword):
    systemmsg = "You are a article image finder for wordpress articles."

    messages = [{"role": "system", "content": systemmsg}]
    messages.append({"role": "user", "content": f"Find an image for this article titled: {keyword} be sure not to serch for the title but for images that might repesent article e.g: News, or Journalist."})
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_and_download",
                "description": "Search and downloads a random image from the search term, only call this function once per message. - May have to input the same exact search term a few times to get the perfect image.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_term": {
                            "type": "string",
                            "description": "The term to search for, e.g., 'news'",
                        },
                    },
                    "required": ["search_term"],
                },
            },
        }
    ]

    #loop through this
    counter = 0
    while True:
        if counter > 5:
            try:
                #generate new image
                messages = list()
                systemmsg = "You are a prompt enegineer for AI generated images."
                messages.append({"role": "system", "content": systemmsg})
                messages.append({"role": "user", "content": f"Generate a prompt for Dall-e to generate an image for {keyword} article. You will have to describe exactly what you want to see to every detail. Dont use IP or trademarked content."})
                dalle_prompt = client.chat.completions.create(
                    model=model,
                    messages = messages,
                )
                prompt = dalle_prompt.choices[0].message.content
                print("\n\nDalle Prompt:",prompt)

                # Generate the image
                response = client.images.generate(
                    model="dall-e-3",
                    prompt=prompt,
                    size="1024x1024",
                    quality="hd",
                    n=1,
                    style="vivid",
                )
                image_url = response.data[0].url
                print("\n\nDalle Image URL:",image_url)
                
                # Download and save the image
                img_data = requests.get(image_url).content
                with open('image.jpg', 'wb') as handler:
                    handler.write(img_data)
                print("Image saved as image.jpg")
                break

            except Exception as err:
                # Handle the error here
                print("Error:", err)
                counter += 1
                continue


        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        response_message = response.choices[0].message
        print("\n\nResponse 1:",response_message.content)
        tool_calls = response_message.tool_calls

        if tool_calls:
            available_functions = {"search_and_download": search_and_download}
            messages.append(response_message)
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions[function_name]
                function_args = json.loads(tool_call.function.arguments)
                function_response = function_to_call(
                    search_term=function_args.get("search_term"),
                    filename=function_args.get("filename", "image.jpg"),
                )
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    }
                )
            second_response = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=messages,
            )
        
            print("\n\nResponse 2:",second_response.choices[0].message.content)
            messages.append(second_response.choices[0].message)

            image_url = json.loads(function_response)["image_url"]
            image_messages = [
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"is this image sutatble for the article titled {keyword}? If not then say no, explain what the image was in one sentence and say try again, you can use the same search term again or a new one if it still isnt working. Note: The image doesnt have to be perfect but it should resemble something in the article."},
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                    },
                    },
                ],
                }
            ]
            
            third_response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=image_messages,
            )

            print("\n\nThird Response: ",third_response.choices[0].message.content)
            messages.append({"role": "user", "content": third_response.choices[0].message.content})

            if "no" in third_response.choices[0].message.content.lower():
                #restart loop
                print("\n\nRestarting loop")
                # print(messages)
                counter += 1
                continue
                
            else:
                #stop loop
                print("\n\nStopping loop because of yes in response")
                # print(messages)
                counter = 0
                break
        else:
            #stop loop
            # print(messages)
            print("\n\nStopping loop because no tool calls")
            counter = 0
            break  




if __name__ == "__main__":
    keyword = "Top Tech Publications"
    run_images(keyword)


#TODO Have it see images that its already used and not use them again. add a general screenshot of the website maybe?