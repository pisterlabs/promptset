import numpy as np
import pandas as pd
import pickle
import random
import openai
import os
import datetime

def predict_crops(nitrogen, phosphorous, potassium, temperature, humidity, ph, rainfall, size):
    model= pickle.load(open('model.pkl','rb'))
    inputs= [nitrogen, phosphorous, potassium, temperature, humidity, ph, rainfall]
    inputs= np.array(inputs).reshape(1,-1)
    probabilities= model.predict_proba(inputs)[0]
    probabilities= pd.Series(probabilities, index= model.classes_)
    max= probabilities.idxmax()
    probabilities.drop(max, inplace=True)
    second_max= probabilities.idxmax()
    best_two= [max, second_max]
    best_two = [str(item) + ' YIELD (Kg per ha)' for item in best_two]
               
    df_yields= pd.read_csv('yields.csv')
    results= df_yields[df_yields['Dist Name']=="Guntur"][best_two]
    crop1= {results.columns[0][:-18]: results.iloc[0,0]}
    crop2= {results.columns[1][:-18]: results.iloc[0,1]}

    revenue= {'CHICKPEA': 0.421, 'RICE':0.265, 'MAIZE':0.14, 'PIGEONPEAS':0.625, 'COTTON':0.687 }

    if list(crop1.keys())[0] == 'CHICKPEA' or list(crop2.keys())[0] == 'CHICKPEA':
        type= 'Mixed'
    else:
        type= random.choice(['Row', 'Strip'])

    result= revenue[list(crop1.keys())[0]]*crop1[list(crop1.keys())[0]] + revenue[list(crop2.keys())[0]]*crop2[list(crop2.keys())[0]]

    mydate = datetime.datetime.now()
    month= mydate.strftime("%B")

    openai.api_key = os.environ.get("API_OPENAI")
    response_whattodonext = openai.Completion.create(
    engine="text-davinci-003",
    prompt="Assume i am a farmer,Can you provide detailed farming advice for the {} of august specifically focusing on {} and {}? Please include the activities, and precautions to be taken during this period. Answer in 80 words.make it breif and pointer".format(month,max,second_max),
    max_tokens=1000
    )

    response_whycr = openai.Completion.create(
    engine="text-davinci-003",
    prompt="Can you explain the benefits of {} intercropping with {} and {} in terms of nutrient synergy and soil health? Please provide a concise description of why this practice is advantageous for these specific crops.".format(type,max,second_max),
    max_tokens=1000
    )

    return {"Revenue":round(result*size/2,2), "Type":type, "Crop1":max,"Crop1 Value": round(crop1[max]/2*size,2) , 
            "Crop2":second_max, "Crop2 Value": round(crop2[second_max]/2*size, 2), 
            "Advice1": response_whattodonext.choices[0].text.strip(), "Advice2": response_whycr.choices[0].text.strip()}
