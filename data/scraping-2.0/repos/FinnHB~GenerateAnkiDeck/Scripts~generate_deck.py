"""
This script uses chatGPT and langchain to generate Anki flashcards for language learning. The script is
designed to create flashcards for Mandarin Chinesese HSK levels, however, with several adjustments, the
script should be able to take in any list of words or phrases in any language and generate an Anki deck.

If the temperature score of the GPT models is greater than 0, results may vary for each iteration. For
the HSK vocabulary, the list of vocabulary is taken from the lemmih's github repo (https://github.com/lemmih),
offering HSK vocabulary from level 1 through to 6.

To use this script, you must have an OpenAI API key.

Date: 01/01/2024
Author: Finn-Henrik Barton
"""


#-------------#
#-- IMPORTS --#
#-------------#
#Basic
import pandas as pd
import os
from tqdm import tqdm

#OpenAI and Langchain functions
from OpenAI_API_Key import api_key
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

#Package for generating Anki decks & custom card model
import genanki
from Scripts.genanki_model_templates import GPT_MODEL


#---------------#
#-- FUNCTIONS --#
#---------------#
def query_gpt(model, query, context=""):
  """
  Queries a GPT model, passing a query and potential context. This function is primarily used for
  generating example sentences for words or phrases.

  Parameters:
  -----------
  model : langchain_community.chat_models.openai.ChatOpenAI
    Langchain model to use for querying
  query : str
    String to pass to the langchain model, for example, "translate 'Hello' to french"
  context : str, default is ""
    Context to pass to the langchain model, for example, "You are a helpful english to french translator"
  """
  #Generate message
  messages = [SystemMessage(content=context), HumanMessage(content=query)]

  #Generate respose, returning only the content
  return model(messages).content



#----------------#
#-- PARAMETERS --#
#----------------#
#-- API key --#
os.environ["OPENAI_API_KEY"] = api_key


#-- Translation Parameters --#
#Basic mandatory fields
hsk_level = 2                                                                             # HSK level, specific to learning Chinese
native_language = "English"                                                               # Language to translate words into
learning_language = "Mandarin Chinese"                                                    # Language to translate the words from

#Model temperatures (from 0 (no variation) to 2 (significant variation))
gpt_sentence_temperature = 1
gpt_translate_temperature = 0

#Additional information or queries
additional_sentence_context = "Try to use simple words in the HSK1 to HSK3 vocabulary lists."
additional_sentence_query = "Include the translated sentence, the pinyin, and english translation, each separated by a new line."


#-- Anki Parameters --#
#Deck ID and names
deck_id = 2059400112
deck_name = f"HSK{hsk_level} (GPT Generated)"
deck_folder = os.path.join(".", "Decks")


#-- Query Templates --#
#Parameters for the GPT model for creating sentences
gpt_translate_context = 'You are a {learning_language} to {native_language} dictionary, providing concise translations. You only return the translation and pinyin.'
gpt_sentence_context = 'You are a helpful assistant that is helping an {native_language} speaker to learn {learning_language}. {additional_sentence_context}'
gpt_sentence_query = 'Create a short example sentence in {learning_language} that uses "{to_translate}". {additional_sentence_query}'


#-- Word List --#
wordlist_file = f"https://raw.githubusercontent.com/lemmih/lesschobo/master/data/HSK_Level_{hsk_level}_(New_HSK).csv"




#----------#
#-- MAIN --#
#----------#
if __name__ == "__main__":
  #-- Initialise --#
  #Initialise GPT models
  translation_model = ChatOpenAI(temperature=gpt_translate_temperature)
  sentence_model = ChatOpenAI(temperature=gpt_sentence_temperature)

  #Initialise anki deck
  deck = genanki.Deck(deck_id, deck_name)


  #-- Read & Format Word List --#
  #Read in word list file
  word_df = pd.read_csv(wordlist_file, index_col=0, header=1)

  #Data-source specific formatting, in this case, removing any words which do not correspond with the HSK level.
  mask =   [str(x)[0] == str(hsk_level) for x in word_df["HSK \nLevel-Order"]]
  words = word_df.loc[mask, "Word"].values



  #-- Definition, Example Sentence, and Add to Deck --#
  for word in tqdm(words):
    #Get defintion
    definition = query_gpt(translation_model,
                          query = word,
                          context=gpt_translate_context.format(learning_language=learning_language,
                                                               native_language=native_language))

    #Get example sentence by populating context and query templates
    example_sentence = query_gpt(sentence_model,
                                query = gpt_sentence_query.format(learning_language=learning_language,
                                                                  to_translate=word,
                                                                  additional_sentence_query=additional_sentence_query),
                                context=gpt_sentence_context.format(learning_language=learning_language,
                                                                    native_language=native_language,
                                                                    additional_sentence_context=additional_sentence_context))

    #Replacing new lines with HTML break marks
    example_sentence = example_sentence.replace("\n", "<br>")


    #Create a note card and add to deck
    note = genanki.Note(model=GPT_MODEL, fields=[word, definition, example_sentence])
    deck.add_note(note)


  #-- Writing --#
  #Write deck
  deck_file = os.path.join(deck_folder, f'{deck_name}.apkg')
  genanki.Package(deck).write_to_file(deck_file)