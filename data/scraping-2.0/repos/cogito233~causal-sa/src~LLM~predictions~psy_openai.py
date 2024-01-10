import os
import openai
import csv
import pandas as pd
import pickle
from efficiency.nlp import Chatbot

c1_0="""As a customer writing a review, I initially composed the following feedback: "{}"
After carefully considering the facts, I selected a star rating from the options of "1", "2", "3", "4", or "5". My final rating was:"""
c1_1="""As a customer sharing my experience, I crafted the following review: "{}"
Taking into account the details of my experience, I chose a star rating from the available options of "1", "2", "3", "4", or "5". My ultimate rating is:"""
c1_2="""As a client providing my opinion, I penned down the subsequent evaluation: "{}"
Upon thorough reflection of my encounter, I picked a star rating among the choices of "1", "2", "3", "4", or "5". My conclusive rating stands at:"""
c1_3="""As a patron expressing my thoughts, I drafted the ensuing commentary: "{}"
After meticulously assessing my experience, I opted for a star rating from the range of "1", "2", "3", "4", or "5". My definitive rating turned out to be:"""
c1_4="""As a consumer conveying my perspective, I authored the following assessment: "{}"
By carefully weighing the aspects of my interaction, I determined a star rating from the possibilities of "1", "2", "3", "4", or "5". My final verdict on the rating is:"""

c2_0="""As a customer writing a review, I initially selected a star rating from the options "1", "2", "3", "4", and "5", and then provided the following explanations in my review: "{}"
The review clarifies why I gave a rating of"""
c2_1="""As a customer sharing my experience, I first chose a star rating from the available choices of "1", "2", "3", "4", and "5", and subsequently elaborated on my decision with the following statement: "{}"
The review elucidates the reasoning behind my assigned rating of"""
c2_2="""As a client providing my opinion, I initially picked a star rating from the range of "1" to "5", and then proceeded to justify my selection with the following commentary: "{}"
The review sheds light on the rationale for my given rating of"""
c2_3="""As a patron expressing my thoughts, I started by selecting a star rating from the scale of "1" to "5", and then offered an explanation for my choice in the following review text: "{}"
The review expounds on the basis for my designated rating of"""
c2_4="""As a consumer conveying my perspective, I began by opting for a star rating within the "1" to "5" spectrum, and then detailed my reasoning in the subsequent review passage: "{}"
The review delineates the grounds for my conferred rating of"""

c0_0="""You are an experienced and responsible data annotator for natural language processing (NLP) tasks. In the following, you will annotate some data for sentiment classification. Specifically, given the task description and the review text, you need to annotate the sentiment in terms of "1" (most negative), "2", "3", "4", and "5" (most positive).
Review Text: "{}"
Sentiment:"""
c0_1="""As a proficient data annotator in natural language processing (NLP), your responsibility is to determine the sentiment of the given review text. Please assign a sentiment value from "1" (very negative) to "5" (very positive).
Review Text: "{}"
Sentiment Score:"""
c0_2="""As a skilled data annotator in the field of natural language processing (NLP), your task is to evaluate the sentiment of the given review text. Please classify the sentiment using a scale from "1" (highly negative) to "5" (highly positive).
Review Text: "{}"
Sentiment Rating:"""
c0_3="""As an expert data annotator for NLP tasks, you are required to assess the sentiment of the provided review text. Kindly rate the sentiment on a scale of "1" (extremely negative) to "5" (extremely positive).
Review Text: "{}"
Sentiment Evaluation:"""
c0_4="""As a proficient data annotator in natural language processing (NLP), your responsibility is to determine the sentiment of the given review text. Please assign a sentiment value from "1" (very negative) to "5" (very positive).
Review Text: "{}"
Sentiment Assessment:"""


def main():
    chat = Chatbot(model_version='gpt3.5', max_tokens=100,system_prompt="You are a helpful assistant.", 
                 openai_key_alias='OPENAI_API_KEY', openai_org_alias='OPENAI_ORG_ID')
    df=pd.read_csv("./test_yelp_1k.csv")
    engine="gpt-3.5-turbo"
    #prompt_groups=[[c0_0,c0_1,c0_2,c0_3,c0_4],[c1_0,c1_1,c1_2,c1_3,c1_4],[c2_0,c2_1,c2_2,c2_3,c2_4]]
    prompt_groups=[[c1_0,c1_1,c1_2,c1_3,c1_4],[c2_0,c2_1,c2_2,c2_3,c2_4]]
    #with open('./'+engine+'_test_1k.csv', 'w') as csvoutput:
    #    writer = csv.writer(csvoutput, lineterminator='\n')
    #    row=['pred','review_id','prompt_type','prompt_id']
    #    writer.writerow(row)
    for j,group in enumerate(prompt_groups):
        for k,prompt in enumerate(group):
            for i in range(0,df.shape[0],1):
                review=df['text'].values[i]
                prompts=prompt.format(review)
                response = chat.ask(prompts,engine=engine)
                dict_res={}
                dict_res['response']=response
                dict_res['review_id']=df['review_id'].values[i]
                dict_res['prompt_type']=j+1
                dict_res['prompt_id']=k
                with open('./'+engine+'_test_1k.csv', 'a') as csvoutput:
                    writer = csv.writer(csvoutput, lineterminator='\n')
                    writer.writerow([dict_res['response'],str(dict_res['review_id']),str(j+1),str(k)])

## call main function
if __name__ == '__main__':
    main()
