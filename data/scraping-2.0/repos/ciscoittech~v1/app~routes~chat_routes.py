from flask import render_template, request
import os
import openai
from dotenv import load_dotenv
from app import app

load_dotenv()
openai.api_key = os.environ["OPEN_API_KEY"]
print(openai.api_key)


messages = [
    {
        "role": "system",
        "content": """TechLife Coach: Welcome to TechLife, your dedicated AI life coach specializing in tech careers! 
             Whether you're a budding programmer, an experienced developer, or someone keen on diving into the world of technology, I'm here to guide you. 

Let's embark on a transformative journey:
- Dive deep into your career aspirations and motivations 🔍
- Chart out your strengths, areas for improvement, and unique selling points 🚀
- Navigate the tech industry's nuances and identify key growth areas 📊
- Create actionable plans, set milestones, and celebrate achievements together 🎉

Begin by sharing your current position in your tech journey and what you hope to achieve. Together, we'll craft a roadmap for success!



Define your specific tech objective or question by entering: /g

Need help or more resources? Just call on me or any other expert agents anytime. We're here to support and amplify your growth!""",
    }
]


@app.route("/coaching", methods=["GET", "POST"])
def coaching():
    if request.method == "POST":
        user_input = request.form["user_input"]
        messages.append({"role": "user", "content": user_input})

        res = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)

        messages.append(res["choices"][0]["message"].to_dict())
        assistant_response = res["choices"][0]["message"]["content"]

        return render_template(
            "home/coaching.html", user_input=user_input, messages=messages
        )

    return render_template("home/coaching.html")
