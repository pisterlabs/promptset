import random
import domains.school.school.library_book as library_book
from action import Action

try:
    import openai
    import game_openai
    ai_installed = True
except ImportError:
    ai_installed = False

title_formats = ['The {adjective1} {noun1}', '{noun1}, {noun2}, and the {adjective1} {noun3}', 
    'How the {noun1} {verb1}èd the {noun2}', 'A tale of {number1} {noun1}s', '{noun1} and {noun2}',
    'The {adjective1} {adjective2} {noun1}', 'A journey to the land of the {adjective1} {noun1}s',
    '{number1} {adjective1} {noun1}: A retrospective', 'On {verb1}ing {noun1}s', 'The {noun1} of {noun2}s']
nouns = ['city', 'plant', 'coffee', 'spell', 'dragon', 'tree', 'kingdom', 'knight', 'castle', 'land', 
'village', 'plain', 'forest', 'marsh', 'hut', 'portal', 'cave', 'mountain', 'empire', 'fire']
adjectives = ['strange', 'old', 'misty', 'crumbling', 'bright', 'dark', 'ancient', 'towering', 'brave', 
'endless', 'unseen', 'distant', 'new', 'dense', 'airy', 'dusty']
verbs = ['conquer', 'capture', 'destroy', 'create', 'perfect', 'combine', 'wash', 'vanquish', 'perfume', 'rescue']
numbers = ['two', 'three', 'fourty-two', 'thirteen', 'eighty-seven', 'fifty-five', 'two hundred fifty']

book_description_adjectives = ['old', 'worn', 'dusty', 'new', 'shiny', 'red', 'orange', 'yellow', 'green', 'blue', 
'violet', 'arcane', 'thin', 'thick', 'tall', 'short', 'yellowed', 'tiny', 'ancient', 'modern', 'strange']
book_messages = ['This page is in a language that you do not understand.', 'This page is mysteriously blank.', 
    'This page is too worn for you to make out the text.', 'This page has been blotted out by an ink spill.']
book_message_weights = [45, 8, 40, 7]

book_styles = ["prose", "poetry"]
book_style_weights = [50, 50]
styling = False

class RandomBook(library_book.LibraryBook):
    def __init__(self):
        t_format = random.choice(title_formats)
        num_nouns = t_format.count('noun')
        num_adjectives = t_format.count('adjective')
        num_verbs = t_format.count('verb')
        num_numbers = t_format.count('number')

        arg_dict = {}
        for i in range(0, num_nouns):
            arg_dict['noun%s' % str(i+1)] = random.choice(nouns)
        for j in range(0, num_adjectives):
            arg_dict['adjective%s' % str(j+1)] = random.choice(adjectives)
        for k in range(0, num_verbs):
            arg_dict['verb%s' % str(k+1)] = random.choice(verbs)
        for l in range(0, num_numbers):
            arg_dict['number%s' % str(l+1)] = random.choice(numbers)

        self.book_title = t_format.format(**arg_dict).title()

        book_adjectives = [random.choice(book_description_adjectives), random.choice(book_description_adjectives)]
        book_s_desc = '%s %s book' % tuple(book_adjectives)
        book_l_desc = 'This is a ' + book_s_desc + ' titled ' + self.book_title + '.'

        self.book_msg = """
    \=============================================
    %s
    \=============================================""" % self.book_title
        self.book_generated = False  # whether the book contents have been auto-generated yet
        self.game.events.call_later(1, self.generate_contents) # generate the book asynchronously

        super().__init__('book', __file__, book_s_desc, book_l_desc)
        self.add_adjectives(*book_adjectives)
        self.set_message(self.book_msg)
    
    async def generate_contents(self):
        if not self.book_generated:  # generate random book contents when called
            global ai_installed
            if ai_installed: 
                try:
                    if styling:
                        book_style = random.choices(book_styles, book_style_weights)[0]
                        ai_body = await game_openai.openai_completion_prompt("You are an author of ancient books found in the library of a text adventure game. Write a book titled %s in the style of %s." % (self.book_title, book_style))
                    else:
                        ai_body = await game_openai.openai_completion_prompt(("You are an author of ancient books found in the library of a text adventure game. Write a book titled %s." % self.book_title))
                    self.book_msg += "\n#*\n#*\n" + ai_body
                except Exception as e:
                    self.log.error(f"Exception when calling OpenAI text completion: {e}")
                    ai_installed = False
            if not ai_installed:
                number_pages = round(random.normalvariate(30, 15))
                page_msg = random.choices(book_messages, book_message_weights)[0]
                for i in range(0, number_pages):
                    self.book_msg += "\n#*\n" + page_msg 
            self.set_message(self.book_msg)
            self.book_generated = True

def clone():
    return RandomBook()

