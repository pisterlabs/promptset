import openai
from dotenv import dotenv_values

config = dotenv_values(".env")

openai.api_key = config["OPENAI_API_KEY"]

exersice = "4"

if (exersice == "1"):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="the top 10 league of legends champions are: ",
        max_tokens=20,
    )

    champions = response.choices[0].text
    print(champions)


# with stop word
if (exersice == "2"):
    response_stop = openai.Completion.create(
        prompt="the top 10 league of legends champions are: ",
        model="text-davinci-003",
        max_tokens=20,
        stop=["4."]
    )

    champions_stop = response_stop.choices[0].text

    print(champions_stop)
    
    
# with n 
if (exersice == "3"):
    response_n = openai.Completion.create(
        prompt="the top 10 league of legends champions are: ",
        model="text-davinci-003",
        max_tokens=20,
        n = 5
    )
    
    for i in range(5):
        champions_n = response_n.choices[i].text
        print(champions_n)
        print("\n")
        
        
# with echo
if (exersice == "4"):
    response_echo = openai.Completion.create(
        prompt="example dict: ",
        model="text-davinci-003",
        max_tokens=20,
        echo = True
    )
    
    champions_echo = response_echo.choices[0].text
    print(champions_echo)
        
