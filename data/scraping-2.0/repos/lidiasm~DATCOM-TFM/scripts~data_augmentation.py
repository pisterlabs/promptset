import re
import openai
import pandas as pd
from textaugment import EDA
from textaugment import Translate
from deep_translator import GoogleTranslator

import nltk  
nltk.download('omw-1.4')


def translate_english_spanish_texts(dataset: pd.DataFrame, text_col: str):
    '''
    Translates english texts to spanish and spanish texts to
    english to increase the population of documents stored
    in a provided dataset. It could be used as a data augmentation
    technique to create more train samples.

    Parameters
    ----------
    dataset : Pandas dataframe
        The data which contains a set of texts to translate.
    text_col : str
        The column name in which the set of texts is stored.

    Returns
    -------
    A list of strings with the translated documents.
    '''
    # Convert the dataset to a list of dictionaries
    dataset_to_dict = dataset.to_dict('records')

    # Initialize two translators
    en_to_es_translator = GoogleTranslator(source='en', target='es')
    es_to_en_translator = GoogleTranslator(source='es', target='en')

    # Variable to save the new texts
    new_texts = []

    # Iterate over the texts to translate them depending on their language
    for record in dataset_to_dict:
        if (record['language'] == 'en'):
            new_texts.append(en_to_es_translator.translate(record[text_col]))

        elif (record['language'] == 'es'):
            new_texts.append(es_to_en_translator.translate(record[text_col]))

    return new_texts


def apply_easy_data_augmentation(dataset: pd.DataFrame, text_col: str, 
                                n_replacements: int, n_times: int):
    '''
    Applies different data augmentation technique to create new samples
    from the provided set of texts.
        * Searchs for N synonums to replace the original words with them.
        * Deletes a specific quantity of words randomly.
        * Swaps the location of some words randomly.
        * Inserts a synonim of a random word in a random position.
    
    All this techniques can be applied the provided number of times.
    ONLY FOR ENGLISH TEXTS.

    Parameters
    ----------
    dataset : Pandas dataframe
        The data which contains a set of texts to augment.
    text_col : str
        The column name in which the set of texts is stored.
    n_replacements : int
        The number of words to replace with their synonyms
    n_times : int
        The number of times to apply this technique. 
        E.g.: 1 time produces the double of samples.
        E.g.: 2 times produces the triple of samples.
        ....

    Returns
    -------
    A list of strings with the original texts along with
    the new synthetic documents.
    '''
    # Initialize an EDA object
    text_aug_obj = EDA()

    # Get texts only
    train_texts = list(dataset[text_col].values)

    # Add the original texts to the final variable
    # in which the new samples will be stored as well
    aug_texts = list(train_texts)

    # Iterate over the texts to create new samples
    for time in range(0, n_times):
        for text in train_texts:
            aug_texts.append(text_aug_obj.synonym_replacement(
                sentence=text, 
                n=n_replacements))
            
            aug_texts.append(text_aug_obj.random_deletion(
                sentence=text, 
                p=0.2))
            
            aug_texts.append(text_aug_obj.random_swap(text))
            
            aug_texts.append(text_aug_obj.random_insertion(text))
    
    return aug_texts


def apply_round_trip_translation(dataset: pd.DataFrame, 
    text_col: str, src_lang: str, targ_lang: str):
    '''
    Applies the data augmentation technique called Round-Trip 
    Translation in which the goal is to create new texts from the
    provided ones translating them to a specific language to then 
    translating them back to the source language.

    Parameters
    ----------
    dataset : Pandas dataframe
        The data which contains a set of texts to augment.
    text_col : str
        The column name in which the set of texts is stored.
    src_lang : str
        The language in which the provided texts are written.
    targ_lang : str
        The language to translate the provided texts to.

    Returns
    -------
    A Pandas dataframe with the original and translated texts
    along with their class labels for 'task1' and 'task2' columns.
    '''
    # Initialize a translator
    trans_obj = Translate(src=src_lang, to=targ_lang)

    # Get the texts of the source language
    filtered_dataset = (dataset[dataset['language'] == src_lang])

    # Save the translated texts and their labels
    translated_texts_labels = {
        text_col: list(filtered_dataset[text_col].values),
        'task1': list(filtered_dataset['task1'].values),
        'task2': list(filtered_dataset['task2'].values)
    }

    # Iterate over the train texts to apply RTT
    for record in filtered_dataset.to_dict('records'):
        translated_texts_labels[text_col].append(trans_obj.augment(record[text_col]))
        translated_texts_labels['task1'].append(record['task1'])
        translated_texts_labels['task2'].append(record['task1'])

    return pd.DataFrame.from_dict(translated_texts_labels)


def create_new_texts_by_removing_words(text: str, one_word: bool=True):
    '''
    Creates a new set of texts based on one provided by removing only 
    one word at a time or two consecutive words.

    Parameters
    ----------
    text : str
        A string that represents the document in which the new
        samples will be based on.
    one_word : bool (optional, default True)
        True to remove one word per iteration, False to remove two
        consecutive words per iteration.

    Returns
    -------
    A list of strings with the new generated texts.
    '''
    words = text.split()

    if (one_word):
        return [re.sub(' +', ' ', text.replace(word, '')) for word in words]
    else:
        return [re.sub(' +', ' ', text.replace(words[index-1], '').replace(words[index], '')) \
                for index in range(1, len(words))]
    

def generate_text_using_gpt3(text: str, lang: str):
    '''
    Calls the OpenAI API to send it a task to generate a text 
    based on the provided one but with negative verbs meaning
    the opposite using the Davinci GPT-3 model.

    Parameters
    ----------
    text : str
        A string with the text to take it as an example to generate
        a synthetic sample with negative verbs.
    lang : str
        A string with the language of the texts. Available are:
        en (English) or es (Spanish).

    Returns
    -------
    A string with the generated text by the Davinci GPT-3 model.
    '''
    # Set up the OpenAI API client
    openai.api_key = OPEN_AI_KEY

    # Generate a response
    task = 'Transform the verbs of this sentence to negative' if lang == 'en' \
        else 'Transforma los verbos de estas frases a negativo'
    
    completion = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"{task}: {text}"
    )

    return completion.choices[0].text