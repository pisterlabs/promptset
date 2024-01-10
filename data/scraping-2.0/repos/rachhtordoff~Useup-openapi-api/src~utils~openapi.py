from longchain.llms import OpenAi
from longchain.prompts import ChatPromptTemplate
from longchain.chat_models import ChatOpenAI
from longchain.chains import LLMChain
from longchain.chains import SimpleSequentialChain


def format_template(json):

    created_template = '''

    Give me a recipe, including the cooking method for the items that I have for {selectedNo} people.
    
    Specify which ingredients will expire soon.
    Specify which ingredients are in the pantry (other items) and Specify which ingredients are needed to be added to a shopping list.
    
    Also supply full cooking instructions. 
    
    The meal should be {selectedCuisine} inspired. 
    
    Please also provide detailed cooking instructions.

    '''
    
    if json.get('selectedCuisine') !=='Pot Luck':
        created_template += ' Focus on a {selectedCuisine} inspired meal.'
    if json.get('selectedCalorie') == 'Yes':
        created_template += ' Please include a calorie breakdown'
    if json.get('selectedDietry'):
        
        created_template += ' Please make sure the recipe is {dietaryString}'

    prompted_template = ChatPromptTemplate.from_template(created_template)

    if json.get('selectedDietry'): 
        dietaryString = ', '.join(json.get('selectedDietry'))
        filled_template = prompted_template.format_messages(
                selectedNo=json.get('selectedNo'),
                selectedCuisine=json.get('selectedCuisine'),
                dietaryString=dietaryString
        )
    else:
        filled_template = prompted_template.format_messages(
            selectedNo=json.get('selectedNo'),
            selectedCuisine=json.get('selectedCuisine'),
        )

    response = chat(filled_template).content

    return response


def format_template_second(json):

    created_template = '''

    Make sure this recipe is in the following format
    
    ingredients will expire soon and are in the recipe have been specified
    ingredients are in the pantry (other items) and are in the recipe have been specified
    any ingredients in the recipe that are not owned yet are displayed in a shopping list.
    
    full cooking instructions have been supplied
    
    '''
    if json.get('selectedCalorie') == 'Yes':
        created_template += ' A full calorie breakdown has been included'

    prompted_template = ChatPromptTemplate.from_template(created_template)


    return prompted_template


def get_chat_response(prompt, prompt2, json):
    chat = OpenAi(temparature=0.0)
    
    chain1 = LLMChain(llm=chat, prompt=prompt)
    chain2 = LLMChain(llm=chat, prompt=prompt2)

    simple_sequential_chain = SimpleSequentialChain(chains=(chain1, chain2), verbose=True)
    simple_sequential_chain.run('can you give me a recipe')
