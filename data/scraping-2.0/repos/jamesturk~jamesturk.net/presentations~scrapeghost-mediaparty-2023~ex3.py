import requests
import openai

def get_details(url):
    response = requests.get(url)

    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": "When provided with HTML, return the equivalent JSON in the format {'name': '', 'position': '', 'hired': 'YYYY-MM-DD'}"},
                {"role": "user", "content": response.text}
            ]
    )
    print(resp.choices[0].message.content)

get_details("https://scrapple.fly.dev/staff/3")
get_details("https://scrapple.fly.dev/staff/4?style=new")
get_details("https://scrapple.fly.dev/staff/5?style=experimental")