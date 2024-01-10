# -*- coding: utf-8 -*-
import openai
from dotenv import load_dotenv
import os
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEYY')
from generalGpt3 import GPT3

prompt_en =   """Sentiment classification is a classification problem that predicts the sentiment of a sentence. \
If a sentence is negative, it will classify it as negative, if it is neutral it will classify it as neutral and \
if it is positive it will classify it as positive.
Here are some examples of sentiment classification:

Sentence: This movie by Steven Spielberg is supergood, I loved it!
Sentiment: Positive

Sentence: I hated that movie
Sentiment: Negative

Sentence: The weather is great today, I think I'll go for a walk!
Sentiment: Positive

Sentence: The movie was just as expected I guess
Sentiment: Neutral

Sentence: I would characterize this movie as a typical science fiction movie
Sentiment: Neutral

Sentence: He screamed and shouted
Sentiment: Negative

Sentence: I love to go for long walks in the forest
Sentiment: Positive

Sentence:"""

prompt_no =   """Sentimentanalyse er en klassifiseringsoppgave som bestemmer stemningen i en setning. \
Hvis en setning er negativ vil den bli klassifisert som negativ, hvis setningen er nøytral blir den klassifisert som nøytral og \
hvis den er positiv blir den klassifisert som positiv.
Under er noen eksempler på sentimentanalyse:

Setning: Denne filmen av Steven Spielberg var veldig bra, jeg elsket den!
Stemning: Positiv

Setning: Jeg hatet den filmen
Stemning: Negativ

Setning: Det er veldig bra vær i dag, jeg tror jeg skal gå ut en tur!
Stemning: Positiv

Setning: Den filmen var akkurat som forventet
Stemning: Nøytral

Setning: Jeg tror jeg vil kategorisere denne filmen som en typisk science fiction film
Stemning: Npytral

Setning: Han ropte og skrek
Stemning: Negativ

Setning: Jeg elsker å gå lange turer i skogen
Stemning: Positiv

Setning:"""

GPT3(prompt_en, "Sentence:", "Sentiment:",temperature=0.7, frequency_penalty=0.8, presence_penalty=0.6)
