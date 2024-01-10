import openai
openai.api_key = "API KEY"
import pandas as pd
import os

# Creazione del DataFrame risultato
result_df = pd.DataFrame(columns=['ID', 'Negative score', 'Neutral score', 'Positive score'])
result_df_noscore = pd.DataFrame(columns=['id', 'sentiment', 'explaination', 'testo'])
df1_sem1_2021 = pd.read_csv('D:\VSCode\Campioni ChatGPT\Risultati con testo\Camp1 Sem1 2021.csv')
df2_sem1_2021 = pd.read_csv('D:\VSCode\Campioni ChatGPT\Risultati con testo\Camp2 Sem1 2021.csv')
df3_sem1_2021 = pd.read_csv('D:\VSCode\Campioni ChatGPT\Risultati con testo\Camp3 Sem1 2021.csv')
df1_sem2_2021 = pd.read_csv('D:\VSCode\Campioni ChatGPT\Risultati con testo\Camp1 Sem2 2021.csv')
df2_sem2_2021 = pd.read_csv('D:\VSCode\Campioni ChatGPT\Risultati con testo\Camp2 Sem2 2021.csv')
df3_sem2_2021 = pd.read_csv('D:\VSCode\Campioni ChatGPT\Risultati con testo\Camp3 Sem2 2021.csv')
df1_sem1_2022 = pd.read_csv('D:\VSCode\Campioni ChatGPT\Risultati con testo\Camp1 Sem1 2022.csv')
df2_sem1_2022 = pd.read_csv('D:\VSCode\Campioni ChatGPT\Risultati con testo\Camp2 Sem1 2022.csv')
df3_sem1_2022 = pd.read_csv('D:\VSCode\Campioni ChatGPT\Risultati con testo\Camp3 Sem1 2022.csv')
df1_sem2_2022 = pd.read_csv('D:\VSCode\Campioni ChatGPT\Risultati con testo\Camp1 Sem2 2022.csv')
df2_sem2_2022 = pd.read_csv('D:\VSCode\Campioni ChatGPT\Risultati con testo\Camp2 Sem2 2022.csv')
df3_sem2_2022 = pd.read_csv('D:\VSCode\Campioni ChatGPT\Risultati con testo\Camp3 Sem2 2022.csv')

df= df2_sem2_2021

for index, row in df.iterrows():
    testo = row['text']
    sentiment = row['sentiment']
    id_originale = str(row['id'])

# Invia il testo a ChatGPT e restituisce solo se Positivo,Negativo o Neutral
    prompt = "Spiega il perchè questo testo : '" + testo +"' sia considerato" + sentiment
    response = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        max_tokens=100,
        temperature=0.5,
        n=1,
        stop=None,
        timeout=10
    )
    print(index)
    explaination = response.choices[0].text.strip()
    result_df_noscore.loc[index] = [id_originale, sentiment,explaination,testo]

# Stampare il DataFrame risultato
#print(result_df_noscore)
os.chdir('DIRECTORY')
result_df_noscore.to_csv("CSV NAME", index = False)
