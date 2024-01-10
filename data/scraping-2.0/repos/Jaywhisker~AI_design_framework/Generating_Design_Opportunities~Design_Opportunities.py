#####################################################################################################
# This is the functions to generate design opportunities based on all the data collected
# Data include: Categorised reviews and comments + summarised youtube transcripts
# imports: openai and helper functions from Helper.

######################################################################################################

import openai
from ...Helper import *

#function to give suggestions on what to improve for each respective category with regards to the product 
#require the keyword, the product name and data which is the list of negative comments related to the category of that product
def suggestion_maker(data, keyword, search_term, model = "text-davinci-003", apikey):
  openai.api_key = apikey
  prompt = "based on the paragraph below, what is the best way to improve the" + " ".join(search_term) + " with regards to " + keyword + "only ?\n" + str(data)
  model = "text-davinci-003"
  suggestion = generate_text(prompt, model)
  return suggestion.strip()


#function to state what should be maintained for each respective category with regards to the product
#require the keyword, the product name and data which is the list of positive comments related to the category of that product
def maintain_maker(data, keyword, search_term, model = "text-davinci-003",apikey):
  openai.api_key = apikey
  prompt = "based on the paragraph below, what is the best way to maintain in the" + " ".join(search_term) + " with regards to " + keyword + "only ?\n" + str(data)
  model = "text-davinci-003"
  maintainence = generate_text(prompt, model, apikey)
  return maintainence.strip()

#function that creates the total_opportunities through GPT-3
#requires: all negative keywords, top 5 posiitve keywords, all categorised data, the summarised yt transcript, search_terms and api key
#the code will extract all negative comments categorised under the negative keywords and ask GPT-3 for suggestions before merging it with the summarised yt transcript
#the code will repeat with the top 5 positive keywords
def reviews_design_outcomes(negative_design_outcomes, positive_design_outcomes, categorical_data, summarised_transcript, search_terms, model = "text-davinci-003",apikey):
  total_opportunities = {} #dictionary to hold all the suggestions (merges features_extractor and suggestion/maintain_maker outputs)

  #negative design outcomes
  for n_outcomes in negative_design_outcomes: #iterate through every negative keyword
    negative_key = n_outcomes[0] #get negative keyword
    negative_comments = categorical_data[negative_key]['negative'] #get negative comments related to category

    #update total_opportunities with suggestions from yt transcript, yt comments, shopee and amazon reviews for negative keywords
    total_opportunities[negative_key] = summarised_transcript[negative_key] + " " + suggestion_maker(negative_comments, negative_key, search_terms, model = "text-davinci-003",apikey)
    print(total_opportunities)


  #positive design outcomes
  for p_outcomes in positive_design_outcomes:
    positive_key = p_outcomes[0] #get positive keyword
    positive_comments = categorical_data[positive_key]['positive'] #get positive comments related to category

    #update total_opportunities on what to maintain from yt transcript, yt comments, shopee and amazon reviews
    total_opportunities[positive_key] = summarised_transcript[positive_key] + " " + maintain_maker(positive_comments, positive_key, search_terms, model = "text-davinci-003",apikey)
    print(total_opportunities)
    
 return total_opportunities


#function that tells GPT-3 to give us a new product specification
#the prompt must mainly contain the reviews (containing things to maintain and suggestions for improvements) as well as prompts to compare with current product specifcations
#after which you can save the file
def generate_design_outcomes(design_outcomes, search_terms, model = "text-davinci-003", apikey):
  openai.api_key = apikey
  prompt = "Imagine you are a product designer and these are the reviews you have received. Using the current " + " ".join(search_terms) + " specifications, provide a new set of product specifications with comparison to the current one to design an improved " + " ".join(search_terms) + " that meets the demands of the reviews. \n Reviews:" + str(design_outcomes)
  final_design_outcomes = generate_text(prompt, model)
  return final_design_outcomes.strip()


