from abc import ABC, abstractmethod
from random import randint
from string import punctuation

from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from json import load

import openai, os


def generate_sentence_prefix():
    all_synonyms = {
        'website': ['site', 'page', 'webpage', 'web page', 'internet site', 'Internet website', 'online page'],
        'create': ['make', 'build', 'construct', 'design', 'produce', 'generate', 'craft', 'synthesize', 'i want',
                   'we need', 'we want', 'give', ''],
        'me': ['for me', 'us', 'our team', 'my team', 'the team', 'the company', 'the business', 'our group', ''],
        'with': ['that contains', 'that has', 'having', 'made of', 'composed of', 'containing', 'where'],
    }
    sentence_base = ['Create', 'me', 'a', 'website', 'with']
    for i, word in enumerate(sentence_base):
        lower_word = word.lower()
        # Special Rule: Remove 'me' when 'Create' is not used
        if lower_word == 'me' and i - 1 >= 0 and (sentence_base[i - 1] == '' or len(sentence_base[i - 1].split()) > 1):
            # Remove 'me'
            sentence_base[i] = ''
            continue
        if lower_word in all_synonyms:
            word_synonyms = all_synonyms[lower_word]
            # Decide which word to use; we subtract 1 because the default word doesn't have an index
            word_to_use = randint(0, len(word_synonyms)) - 1
            if word_to_use > -1:
                sentence_base[i] = word_synonyms[word_to_use]

    # Capitalize first word of sentence, most of the time
    if sentence_base[0] and randint(0, 5) > 0: # changed because it is giving errors
        sentence_base[0] = sentence_base[0][0].upper() + sentence_base[0][1:]

    return ' '.join(sentence_base)


def get_readable_relative_position(position):
    text_position = ''

    # Corner positions
    if position['horizontal'] == -2 and position['vertical'] == -2:
        text_position += 'top left corner'
    elif position['horizontal'] == -1 and position['vertical'] == -2:
        text_position += 'top and left center'
    elif position['horizontal'] == 0 and position['vertical'] == -2:
        text_position += 'top center'
    elif position['horizontal'] == 1 and position['vertical'] == -2:
        text_position += 'top and right center'
    elif position['horizontal'] == 2 and position['vertical'] == -2:
        text_position += 'top right corner'
    elif position['horizontal'] == -2 and position['vertical'] == -1:
        text_position += 'left edge and above center'
    elif position['horizontal'] == -2 and position['vertical'] == 0:
        text_position += 'left edge and vertical center'
    elif position['horizontal'] == -2 and position['vertical'] == 1:
        text_position += 'left edge and below center'
    elif position['horizontal'] == 2 and position['vertical'] == -1:
        text_position += 'right edge and above center'
    elif position['horizontal'] == 2 and position['vertical'] == 0:
        text_position += 'right edge and vertical center'
    elif position['horizontal'] == 2 and position['vertical'] == 1:
        text_position += 'right edge and below center'
    elif position['horizontal'] == -2 and position['vertical'] == 2:
        text_position += 'bottom left corner'
    elif position['horizontal'] == -1 and position['vertical'] == 2:
        text_position += 'bottom and left center'
    elif position['horizontal'] == 0 and position['vertical'] == 2:
        text_position += 'bottom center'
    elif position['horizontal'] == 1 and position['vertical'] == 2:
        text_position += 'bottom and right center'
    elif position['horizontal'] == 2 and position['vertical'] == 2:
        text_position += 'bottom right corner'
    # Edge positions (non-corner)
    elif position['horizontal'] == -1 and position['vertical'] == -1:
        text_position += 'above and to the left of center'
    elif position['horizontal'] == 0 and position['vertical'] == -1:
        text_position += 'above and center-left of center'
    elif position['horizontal'] == 1 and position['vertical'] == -1:
        text_position += 'above and to the right of center'
    elif position['horizontal'] == -1 and position['vertical'] == 1:
        text_position += 'below and to the left of center'
    elif position['horizontal'] == 0 and position['vertical'] == 1:
        text_position += 'below and center-left of center'
    elif position['horizontal'] == 1 and position['vertical'] == 1:
        text_position += 'below and to the right of center'
    elif position['horizontal'] == -1 and position['vertical'] == 0:
        text_position += 'left of center'
    elif position['horizontal'] == 1 and position['vertical'] == 0:
        text_position += 'right of center'
    # center position
    elif position['horizontal'] == 0 and position['vertical'] == 0:
        text_position += 'center'
    return text_position


# The TextGenerator Abstract Base Class serves as the skeleton, or parent, for all text generation subclasses. On
# initialization, any subclass of this class will open and load the metadata from the specified file path
# (see example below).
class TextGenerator(ABC):
    def __init__(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f: #fixed using utf-8
            self.metadata = load(f)
    @abstractmethod
    def generate(self):
        pass


# This text generator subclass uses nltk to remove proper nouns and nearby stop words from descriptions.
# You can create your own text generator subclasses and modify this one.
# Each subclass should have a specific function that is described in comments.
class NoProperNounsTextGenerator(TextGenerator):
    # From: https://www.geeksforgeeks.org/python-text-summarizer/
    def generate(self):
        # sanity checks
        if ('desc' not in self.metadata) or (self.metadata['desc'] == ""):
            return ''
        # Tokenizing the text
        stopWords = set(stopwords.words('english'))
        words = word_tokenize(self.metadata['desc'])

        # Storing proper nouns so we can remove them
        proper_noun_indexes = []
        i = 0
        for (word, tag) in pos_tag(words):
            if tag == 'NNP' or tag == 'NNPS': # added plural proper nouns
                proper_noun_indexes.append(i)
            i += 1

        # Storing stop words that sit next to proper nouns so we can remove them
        stop_word_removal_indexes = []
        for idx in proper_noun_indexes:
            if idx < len(words) - 1 and words[idx + 1] in stopWords:
                stop_word_removal_indexes.append(idx + 1)
            if idx > 0 and words[idx - 1] in stopWords:
                stop_word_removal_indexes.append(idx - 1)

        removal_indexes = sorted(proper_noun_indexes + stop_word_removal_indexes, reverse=True)
        for idx in removal_indexes:
            if idx < len(words):
                words.pop(idx)

        stripped_text = ''.join([' ' + word if word not in punctuation else word for word in words])[1:]

        capitalized_sentences = ''
        for sentence in sent_tokenize(stripped_text):
            capitalized_sentences += sentence.capitalize() + ' '
        return capitalized_sentences[:-1]

        
# Create a subclass that describes the location of objects on the screen (i.e. buttons, images, inputs, iframes) and
# colors used.
class ObjectLocationTextGenerator(TextGenerator):
    def generate(self):
        # Process images
        # A list to hold description of images
        images = []
        # pass websites without images
        if 'images' not in self.metadata:
            pass
        else:
            # loop through each image
            for image in self.metadata['images']:
                # Skip images that are not on screen
                if not image['is_displayed']:
                    continue
                # The description of image should not be empty
                if 'alt' in image and image['alt'] != '':
                    images.append('an image of ' + image['alt'] + ' in the ' + get_readable_relative_position(image['position']))
        
        # process buttons
        buttons = []
        if 'buttons' not in self.metadata:
            pass
        else:
            for button in self.metadata['buttons']:
                # Skip buttons that are not on screen
                if not button['is_displayed']:
                    continue                
                if 'alt' in button and button['alt'] != '':
                    buttons.append('an ' + button["bg-color"] + ' colored button of ' + button['alt'] + ' in the ' + get_readable_relative_position(button['position']))
                elif 'text' in button and button['text'] != '':
                    buttons.append('an ' + button["bg-color"] + ' colored button of ' + button['text'] + ' in the ' + get_readable_relative_position(button['position']))
        
        inputs = []
        if 'inputs' not in self.metadata:
            pass
        else:
            for input in self.metadata['inputs']:
                # Skip inputs that are not on screen
                if not input['is_displayed'] or input['type'] == 'hidden':
                    continue
                if 'desc' in input and input['desc'] != '':
                    inputs.append('an input field of ' + input['desc'] + ' in the ' + get_readable_relative_position(input['position']))

        iframes = []
        if 'iframes' not in self.metadata:
            pass
        else:
            for iframe in self.metadata['iframes']:
                # Skip inputs that are not on screen
                if not iframe['is_displayed']:
                    continue
                if 'title' in iframe and iframe['title'] != '':
                    if not iframe['is_video']:
                        iframes.append('a non-video iframe of ' + iframe['title'] + ' in the ' + get_readable_relative_position(iframe['position']))
                    else:
                        iframes.append('a video iframe of ' + iframe['title'] + ' in the ' + get_readable_relative_position(iframe['position']))

        return images, buttons, inputs, iframes

# Create a subclass that describes navigation options (see "navbar" in the json).
class NavDescriptionTextGenerator(TextGenerator):
    def generate(self):
        # check if navbar is not present or empty
        if ('navbar' not in self.metadata):
            return []
        # convert navbar options to list
        if '\n' in self.metadata['navbar']: # if \n appears, it is the delimiter 
            options = self.metadata['navbar'].split("\n")
        else: # otherwise a space is delimiter
            options = word_tokenize(self.metadata['navbar'])
        if '' in options:
            options = options.remove('')
        return options

# Create a subclass that summarizes the page based on text content and description metadata (and other metadata if it
# works well). Feel free to use any public NLP APIs, as long as they are free or very affordable (< $0.01 per 500
# words).
class ContentSummaryTextGenerator(TextGenerator):
    def __init__(self, file_path):
        super().__init__(file_path)
        self.file_path = file_path
        
    def generate(self, words=None):
        desc_text = ''
        if ('desc' in self.metadata) and (self.metadata['desc'] != ''):
            desc_text_generator = NoProperNounsTextGenerator(self.file_path)
            desc_text = desc_text_generator.generate()
        
        navbar = ''
        if ('navbar' in self.metadata) and (self.metadata['navbar'] != ''):
            navbar_desc_generator = NoProperNounsTextGenerator(self.file_path)
            navbar_desc = navbar_desc_generator.generate()
                
        desc_text_generator = ObjectLocationTextGenerator(self.file_path)
        images, buttons, inputs, iframes = desc_text_generator.generate()
                
        
        messages = [
                {"role": "system", "content": "You are writing descriptions of landing pages so a software consulting firm can use those descriptions to build a landing page."},
                {"role": "user", "content": "I will first give you a description of the landing page. Then I will give you the description and locations of objects on the landing page step by step."},
                {"role": "assistant", "content": "Okay."},
                {"role": "user", "content": "The following text is the description of the landing page: " + desc_text},
                {"role": "assistant", "content": "I got the description."},
                {"role": "user", "content": "The following text contains the options of navigation bar: " + navbar},
                {"role": "assistant", "content": "I got the options in the navigation bar."},
                {"role": "user", "content": "The following text describes the images: " + '.'.join(images)},
                {"role": "assistant", "content": "I got images."},
                {"role": "user", "content": "The following text describes the buttons: " + '.'.join(buttons)},
                {"role": "assistant", "content": "I got buttons."},
                {"role": "user", "content": "The following text describes the input fields: " + '.'.join(inputs)},
                {"role": "assistant", "content": "I got input fields."},
                {"role": "user", "content": "The following text describes the iframes: " + '.'.join(iframes)},
                {"role": "assistant", "content": "I got iframes."},
                {"role": "user", "content": "Share with me a generic description of this landing page as if you were explaining to a software consulting firm what you want them to build. Include what they are and where they should be. "},
        ]
        
        # Replace OPENAI_API_KEY with your own secret API key envrionment variable
        # Or you can use the following line to set your API key
        # openai.api_key = "YOUR_API_KEY"
        openai.api_key = os.environ.get('OPENAI_API_KEY')
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages= messages,
            temperature=0,
            max_tokens=words
        )
        
        summary = response.choices[0].message.content
        
        return summary



def get_best_examples():
    """
    Function to get the best examples that from the dataset
    Best examples are the ones that have all the generators returing non-empty values
    """
    for i in range(1000):
        desc_generator = NoProperNounsTextGenerator(f'./data/{str(i+1).zfill(4)}/metadata.json')
        desc = desc_generator.generate()
        navbar_generator = NavDescriptionTextGenerator(f'./data/{str(i+1).zfill(4)}/metadata.json')
        navbar = navbar_generator.generate()
        objects_generator = ObjectLocationTextGenerator(f'./data/{str(i+1).zfill(4)}/metadata.json')
        images, buttons, inputs, iframes = objects_generator.generate()
        if (desc and navbar and images and buttons and inputs and iframes):
            print(f'Metadata {str(i+1).zfill(4)} has everything')
        


    
def test_generators(generator, num_examples=1000, example_id = 0):
    """
    Function to test the generators EXCEPT for the ContentSummaryTextGenerator
    It prints the output of the selected generator on the console.
    Args:
        generator (TextGenerator): various classes of TextGenerator with different functionalities. 
        num_examples (int, optional): controls how many example metadata to use. Defaults to 1000.
        example_id (int, optional): if provided as non-zero, only test the result on one example. Defaults to 0.
    """
    if example_id:
        x = generator(f'./data/{str(example_id).zfill(4)}/metadata.json')
        print(x.generate())
    else:
        for i in range(num_examples):
            x = generator(f'./data/{str(i+1).zfill(4)}/metadata.json')
            print(x.generate())

def test_summarization(example_id, words = None):
    """
    Test summary generator

    Args:
        example_id (int): id of the example to test
        words (int, optional): max tokens that GPT can return. Defaults to None puts no limits.
    """
    x = ContentSummaryTextGenerator(f'./data/{str(example_id).zfill(4)}/metadata.json')
    print(x.generate(words))


if __name__ == '__main__':
    # get_best_examples()
    test_summarization(333)

