import pandas as pd
import openai
import os

path = "/home/weberam2/Dropbox/Personal_Misc/ChatGPT/Jukebox/JoF_filled2.xlsx"

df = pd.read_excel(path)

openai.api_key = os.environ["OPENAI_API_KEY"]

## Albums

albumnoentry = df.loc[pd.isna(df['year']), :].index

for album in albumnoentry:
    prompt = "What album does the song " + df['Song'][album] + " by the artist " + df['Artist.1'][album] + " come from? Answer in the form: The album is called:' without quotes "
    model = "text-davinci-003"
    #model = "gpt-3.5-turbo"
    print(prompt)
    response = openai.Completion.create(engine=model, prompt=prompt, max_tokens=30)
    generated_text = response.choices[0].text
    print(generated_text)
    albumguess = generated_text.split(":",1)[1]
    albumguess = albumguess.strip()
    print(albumguess)
    df['Album'][album] = albumguess

# Year

yearnoentry = df.loc[pd.isna(df['year']), :].index

for year in yearnoentry:
    prompt = "What year did the album " + df['Album'][year] + " by the artist " + df['Artist.1'][year] + " come out? Answer in the form: 'The year was:' without quotes "
    model = "text-davinci-003"
    #model = "gpt-3.5-turbo"
    print(prompt)
    response = openai.Completion.create(engine=model, prompt=prompt, max_tokens=50)
    generated_text = response.choices[0].text
    print(generated_text)
    if ":" in generated_text:
        yearguess = generated_text.split(":",1)[1]
    elif "was " in generated_text:
        yearguess = generated_text.split("was ",1)[1]
    yearguess = yearguess.strip()
    print(yearguess)
    df['year'][year] = yearguess


genrenoentry = df.loc[pd.isna(df['genre']), :].index

for genre in genrenoentry:
    prompt = "What genre is the album " + df['Album'][genre] + " by the artist " + df['Artist.1'][genre] + "? Answer in the form: 'The genre is:' without quotes "
    model = "text-davinci-003"
    #model = "gpt-3.5-turbo"
    print(prompt)
    response = openai.Completion.create(engine=model, prompt=prompt, max_tokens=50)
    generated_text = response.choices[0].text
    print(generated_text)
    if ":" in generated_text:
        genreguess = generated_text.split(":",1)[1]
    elif "is " in generated_text:
        genreguess = generated_text.split("was ",1)[1]
    genreguess = genreguess.strip()
    print(genreguess)
    df['genre'][genre] = genreguess

df.to_excel(path) 