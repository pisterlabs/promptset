"""
This is a page to allow the user to upload their bar 
inventory to be used in the cocktail creation process
"""

# Initial imports
import openai
import pandas as pd
import os
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from dotenv import load_dotenv

# Import the services and utils
from utils.inventory_functions import InventoryService
from utils.image_utils import generate_image
from utils.cocktail_functions import RecipeService
from utils.chat_utils import ChatService, Context

# Load the environment variables
load_dotenv()

# Get the OpenAI API key and org key from the .env file
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORG")

# Define the page config
st.set_page_config(page_title="BarKeepAI", page_icon="./resources/cocktail_icon.png", initial_sidebar_state="collapsed")



# Initialize the session state
def init_inventory_session_variables():
    # Initialize session state variables
    session_vars = [
        'inventory_page', 'inventory_csv_data', 'df', 'inventory_list', 'image_generated', 'context', 'ni_ingredients', 'total_inventory_ingredients_cost',
        'inventory_service', 'recipe_service', 'chat_service'
    ]
    default_values = [
        'get_inventory_choice', [], pd.DataFrame(), [], False, None, [], [], InventoryService(), RecipeService(), ChatService()
    ]

    for var, default_value in zip(session_vars, default_values):
        if var not in st.session_state:
            st.session_state[var] = default_value

init_inventory_session_variables()

def get_inventory_choice():
    st.markdown("#### Documentation notes:")
    st.success('''
           **This is the beginning of the inventory aware cocktail generation functionality.  Currently I am just allowing for the use
           of the dummy inventory, but want to add the ability to allow the user to upload their own inventory.**
            ''')
    st.markdown('---')
    # Instantiate the InventoryService class
    inventory_service = st.session_state.inventory_service
    # Allow the user to choose either to upload their own inventory or use the default inventory
    st.markdown('''<div style = text-align:center>
    <h3 style = "color: black;">Choose your inventory</h3>
                </div>''', unsafe_allow_html=True)
    inventory_choice = st.selectbox('Inventory Choice', ['Use the default inventory'], index = 0, key = 'inventory_choice')
    choice_submit_button = st.button('Submit', use_container_width=True, type = 'primary')
    if choice_submit_button:
        if inventory_choice == 'Upload my own inventory':
            st.session_state.inventory_page = 'upload_inventory'
            st.experimental_rerun()
        else:
            # If the user chooses to use the default inventory, we will load the default inventory
            # from the resources folder
            inventory_service.process_and_format_file(uploaded_file=None)
            st.session_state.inventory_page = 'choose_spirit'
            st.experimental_rerun()

# Create the function to allow the user to upload their inventory.  We will borrow this from the "Inventory Cocktails" page
def upload_inventory():
    # Instantiate the InventoryService class
    inventory_service = st.session_state.inventory_service
    # Set the page title
    st.markdown('''
    <div style = text-align:center>
    <h3 style = "color: black;">Upload your inventory</h3>
    <h5 style = "color: #7b3583;">The first column of your file should contain the name of the spirit\
        and the second column should contain the quantity of that spirit that you have in stock.\
        By uploading your inventory, you are allowing the model to prioritize using up ingredients you already have on hand when suggesting cocktails.</h5>
    </div>
    ''', unsafe_allow_html=True)
    # Create a file uploader
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'xls'])
    if uploaded_file:
        upload_file_button = st.button('Upload File and Continue to Cocktail Creation', use_container_width=True, type = 'secondary')
        # If the user clicks the upload file button, process the file
        if upload_file_button:
            with st.spinner('Converting and formatting your file...'):
                # Process the file, format it, and save it to the redis database
                st.session_state.df = inventory_service.process_and_format_file(uploaded_file)
                # Insert a "Use in Cocktail" column as the first column and set it to False for all rows
                st.session_state.inventory_page = "choose_spirit"
                st.experimental_rerun()
    else:
        st.warning('Please upload a file to continue')

# Define a function that will allow the user to select the spirit they want to use in their cocktail
# We will use the new st.data_editor library to allow for dynamic display of the inventory dataframe
# and the ability to let the user interact with it and select the spirit from their inventory
def choose_spirit():
    st.markdown("#### Documentation notes:")
    st.success('''
           **The user can choose the spirit from the inventory that they would like to feature in their cocktail.  In this mode,
            the model will prioritize using other ingredients from the inventory to minimize the need to bring in outside items,
            but is prompted not to do so if it would compromise the quality of the cocktail.**
            ''')
    st.markdown('---')
    # Instantiate the InventoryService class
    inventory_service = st.session_state.inventory_service
    inventory = inventory_service.inventory
    # Create a dataframe from the inventory dictionary
    inventory_df = pd.DataFrame.from_dict(inventory, orient="columns")
    # Insert a "Use in Cocktail" column as the first column and set it to False for all rows
    inventory_df.insert(0, "Use in Cocktail", False)


    # Set the page title
    st.markdown("##### Choose your spirit")
    st.markdown('**Select the spirit from the inventory below\
                that you are trying to use up. Besides the primary spirit you select,\
                the model will prioritize other items already in your inventory to minimize\
                the need to bring in outside liquors.**')
            
    st.text("")

    edit_df = st.data_editor(
        inventory_df,
        column_config={
        "Use in Cocktail": st.column_config.CheckboxColumn(
            "Use in Cocktail?",
            help="Check this box if you want to use this spirit in your cocktail",
            default=False,
        )
    },
    disabled=["widgets"],
    hide_index=True,
    key="data_editor"
)
    # Let the user know that they can edit the values in the dataframe before submitting
    st.warning("**You can edit the values in the inventory to experiment, including the\
                   spirit names as long as it is in the same format and is a valid liquor.\
               To sort the values, click on the column name.**")
    
    # Check to make sure only one of the "Use in Cocktail" checkboxes is checked, otherwise display an error message
    if edit_df['Use in Cocktail'].sum() == 1:
        display_data_editor_button = st.button(f'Create a cocktail using {edit_df[edit_df["Use in Cocktail"] == True]["Name"].values[0]}', use_container_width=True, type = 'primary')
        if display_data_editor_button:
              # Create an inventory list that is the first column of the dataframe
            inventory_list = edit_df['Name'].tolist()
            st.session_state.inventory_list = inventory_list
            # Get the name of the spirit the user selected
            st.session_state['chosen_spirit'] = edit_df[edit_df['Use in Cocktail'] == True]['Name'].values[0]
            # Set the demo_page session_state variable to the name of the spirit
            st.session_state.inventory_page = "create_cocktail"
            st.experimental_rerun()
    else:
        st.error('Please select exactly one spirit to use in your cocktail')

# Create the function to allow the user to create their cocktail.  We will be repurposing the existing "Create Cocktail" page

def create_cocktail():
    st.markdown("#### Documentation notes:")
    st.success('''
           **This is very similar to the main create cocktail feature, but calls the function that prompts
            the model to prioritize inventory items instead of the main cocktail generating function.**
            ''')
    st.markdown('---')
    # Instantiate the RecipeService class
    recipe_service = st.session_state.recipe_service
    # Build the form 
    # Create the header
    st.markdown('''<div style="text-align: center;">
    <h4 style = "color:#7b3583;">Tell us about the cocktail you want to create!</h4>
    </div>''', unsafe_allow_html=True)
    st.text("")

    # Display a message with the spirit the user selected
    st.markdown(f'''<div style="text-align: center;">
    <h5>Spirit selected: <div style="color:red;">{st.session_state.chosen_spirit}</div></h5>
    </div>''', unsafe_allow_html=True)

    st.text("")

    # Set the chosen_liquor variable to the spirit the user selected
    chosen_liquor = st.session_state.chosen_spirit

    # Allow the user to choose what type of cocktail from "Classic", "Craft", "Standard"
    cocktail_type = st.selectbox('What type of cocktail are you looking for?', ['Classic', 'Craft', 'Standard'])
    # Allow the user the option to select a type of cuisine to pair it with if they have not uploaded a food menu
    cuisine = st.selectbox('What type of cuisine, if any, are you looking to pair it with?', ['Any', 'Fresh Northern Californian', 'American',\
                            'Mexican', 'Italian', 'French', 'Chinese', 'Japanese', 'Thai', 'Indian', 'Greek', 'Spanish', 'Korean', 'Vietnamese',\
                            'Mediterranean', 'Middle Eastern', 'Caribbean', 'British', 'German', 'Irish', 'African', 'Moroccan', 'Nordic', 'Eastern European',\
                            'Jewish', 'South American', 'Central American', 'Australian', 'New Zealand', 'Pacific Islands', 'Canadian', 'Other'])
    # Allow the user to enter a theme for the cocktail if they want
    theme = st.text_input('What theme, if any, are you looking for? (e.g. "tiki", "holiday", "summer", etc.)', 'None')

    # Allow the user to select the GPT model to use
    model = st.selectbox('Which model would you like to use?', ['gpt-3.5', 'gpt-4'])

    # Create the submit button
    cocktail_submit_button = st.button(label='Create your cocktail!')
    if cocktail_submit_button:
        with st.spinner('Creating your cocktail recipe.  This may take a minute...'):
            recipe = recipe_service.get_inventory_cocktail_recipe(st.session_state.inventory_list, chosen_liquor, cocktail_type, cuisine, theme, model)
            if recipe:
                st.session_state.image_generated = False
                st.session_state.inventory_page = "display_recipe"
                st.experimental_rerun()


def display_recipe():
    st.markdown("#### Documentation notes:")
    st.success('''
           **Here we display the generated inventory recipe.  The user is then given multiple options to chat about the recipe, cost out
            the recipe, or generate a training guide for the recipe.**
            ''')
    st.markdown('---')
    # Instantiate the RecipeService class
    recipe_service = st.session_state.recipe_service    
    chat_service = st.session_state.chat_service
    # Load the recipe
    recipe = recipe_service.recipe
    # Create the header
    st.markdown('''<div style="text-align: center;">
    <h4>Here's your recipe!</h4>
    <hr>    
    </div>''', unsafe_allow_html=True)
    # Create 2 columns, one to display the recipe and the other to display a generated picture as well as the buttons
    col1, col2 = st.columns([1.5, 1], gap = "large")
    with col1:
        # Display the recipe name
        st.markdown(f'**Recipe Name:** {recipe.name}')
        # Display the recipe ingredients
        st.markdown('**Ingredients:**')
        # If there are inventory ingredients, display them in red
        for ingredient in recipe.ingredient_names:
                # If the ingredient is in the inventory, display it in red
            if ingredient in st.session_state.inventory_list:
                # Get the index of the ingredient in the ingredient_names list
                index = recipe.ingredient_names.index(ingredient)
                # Display the ingredient in red
                st.markdown(f'* <div style="color:red;">{recipe.ingredients_list[index]}</div>', unsafe_allow_html=True)
            else:
                index = recipe.ingredient_names.index(ingredient)
                # Display the ingredient in black
                st.markdown(f'* {recipe.ingredients_list[index]}')
        # Let the user know the key to the colors
        st.markdown('<div style="color: red;">* Red ingredients are ones that you have in your inventory.</div>', unsafe_allow_html=True)
        
        st.text("")
        
        # Display the recipe instructions
        st.markdown('**Instructions:**')
        for instruction in recipe.instructions:
            st.markdown(f'* {instruction}')
        # Display the recipe garnish
        if recipe.garnish != "":
            st.markdown(f'**Garnish:**  {recipe.garnish}')
        # Display the recipe glass
        if recipe.glass != "":
            st.markdown(f'**Glass:**  {recipe.glass}')
        # Display the flavor profile if there is one
        if recipe.flavor_profile != "":
            st.markdown(f'**Flavor Profile:**  {recipe.flavor_profile}')

    with col2:
        # Display the recipe name
        st.markdown(f'<div style="text-align: center;">{recipe.name}</div>', unsafe_allow_html=True)
        st.text("")
        # Placeholder for the image
        image_placeholder = st.empty()
        # Check if the image has already been generated
        if st.session_state.image_generated == False:
            image_placeholder.text("Generating cocktail image...")
            # Generate the image
            image_prompt = f'A cocktail named {recipe.name} in a {recipe.glass} glass with a {recipe.garnish} garnish'
            st.session_state.image = generate_image(image_prompt)
            st.session_state.image_generated = True
        # Update the placeholder with the generated image
        image_placeholder.image(st.session_state.image['output_url'], use_column_width=True)
        # Markdown "AI image generate by [StabilityAI](https://stabilityai.com)"]"
        st.markdown('''<div style="text-align: center;">
        <p>AI cocktail image generated using the Stable Diffusion API by <a href="https://deepai.org/" target="_blank">DeepAI</a></p>
        </div>''', unsafe_allow_html=True)
        st.warning('**Note:** The actual cocktail may not look exactly like this!')

        # Create an option to chat about the recipe
        chat_button = st.button('Questions about the recipe?  Click here to chat with a bartender about it.', type = 'primary', use_container_width=True)
        if chat_button:
            chat_service = st.session_state.chat_service
            chat_service.initialize_chat(context=Context.RECIPE)
            st.session_state.context = Context.RECIPE
            st.session_state.bar_chat_page = "display_chat"
            switch_page('Cocktail Chat')

        # Create an option to cost out the recipe
        cost_button = st.button('Calculate the cost and potential profit of this recipe.', type = 'primary', use_container_width=True)
        if cost_button:
            recipe_service.cost_recipe()
            recipe_service.get_total_drinks()
            st.session_state.inventory_page = "display_cost"
            st.experimental_rerun()

        # Create an option to generate a training guide
        training_guide_button = st.button('Click here to generate a training guide for this recipe.', type = 'primary', use_container_width=True)
        if training_guide_button:
            # Generate the training guide
            st.session_state.is_inventory_recipe = True
            training_guide = recipe_service.generate_training_guide()
            st.session_state.training_guide = training_guide
            switch_page('Training')
            st.experimental_rerun()

         # Create an option to get a new recipe
        new_recipe_button = st.button('Get a new recipe', type = 'primary', use_container_width=True)
        if new_recipe_button:
            # Clear the session state variables
            st.session_state.image_generated = False
            st.session_state.cocktail_page = "get_cocktail_type"
            # Clear the recipe and chat history
            chat_service.chat_history = []
            recipe_service.recipe = None
            st.experimental_rerun()

# Define a function to display the cost of the recipe -- we will use the RecipeService class to do this
# Create a function to display the cost of the recipe
def display_cost():
    st.markdown("#### Documentation notes:")
    st.success('''
           **This feature pulls the cost of the amount of the inventory ingredients from the inventory data,
            and then passes the rest of the ingredients to an LLM to estimate the cost.  The total cost is then
            utilized to present cost information to the user.**
            ''')
    st.markdown('---')
    chat_service = st.session_state.chat_service
    recipe_service = st.session_state.recipe_service
    recipe = recipe_service.recipe
    inventory_service = st.session_state.inventory_service
    inventory = inventory_service.inventory
    # Create two columns -- one two display the recipe text and the cost per recipe, the other to display the profit
    col1, col2 = st.columns(2, gap = 'medium')
    with col1:
        # Display the recipe name
        st.markdown(f'**Recipe Name:** {recipe.name}')
        # Display the recipe ingredients
        st.markdown('**Ingredients:**')
        # Check to see if the name of each ingredient is in the inventory dataframe regardless of case, and if it is, display it in red
        # If they are not in the inventory dataframe, display them in black
        for ingredient in st.session_state.total_inventory_ingredients_cost:
            # Display the ingredient 
            st.markdown(f'* {ingredient[0]}: {float(ingredient[1]):.1f} {ingredient[2]} = ${float(ingredient[3]):.2f}')
        for ingredient in st.session_state.ni_ingredients:
            st.markdown(f'* {ingredient[0]}: {ingredient[1]} {ingredient[2]} ')
        # Display the total cost of the recipe
        st.session_state.total_cocktail_cost = st.session_state.total_inv_cost + (st.session_state.total_ni_cost/4)
        st.markdown(f'**Total Cost of inventory ingredients:** ${st.session_state.total_inv_cost:.2f}')
        st.markdown(f'**Total Cost of non-inventory ingredients:** ${st.session_state.total_ni_cost/4:.2f}')
        st.markdown(f'**Total Cost of recipe:** ${(st.session_state.total_cocktail_cost):.2f}')

        
    with col2:
        # Calculate and display total costs and the potential profit
        st.markdown(f'**Total cost to use up the amount of {st.session_state.chosen_spirit} in your inventory:**')
        st.markdown(f'You can make **{int(st.session_state.total_drinks)}** of the "{recipe.name}" with the amount of {st.session_state.chosen_spirit} you have in your inventory.')

        total_drinks_cost = st.session_state.total_cocktail_cost * st.session_state.total_drinks
        st.write(f'The total cost of the recipe for {int(st.session_state.total_drinks)} drinks is ${total_drinks_cost:.2f}.')
        # Display the potential profit
        st.markdown('**Potential Profit:**')
        # Create a slider that allows the user to select the price of the drink they want to sell it for
        st.write('Select the price you want to sell the drink for:')
        price = st.slider('Price', min_value=10, max_value=20, value=10, step=1)

        # Calculate the profit
        total_profit = (st.session_state.total_drinks * price) - total_drinks_cost

        # Profit per drink
        profit_per_drink = price - st.session_state.total_cocktail_cost

        # Display the profit
        st.write(f'The total profit for {int(st.session_state.total_drinks)} drinks is ${total_profit:.2f}.')
        st.write(f'The profit per drink is ${profit_per_drink:.2f} or {(profit_per_drink / price) * 100:.2f}%.')

         # Create an option to get a new recipe
        new_recipe_button = st.button('Get a new recipe', type = 'primary', use_container_width=True)
        if new_recipe_button:
            # Clear the session state variables
            st.session_state.image_generated = False
            st.session_state.cocktail_page = "get_cocktail_type"
            # Clear the recipe and chat history
            chat_service.chat_history = []
            recipe_service = RecipeService()
            switch_page('Create Cocktails')
            st.experimental_rerun()

    st.text("")
    st.text("")
    
    # Set the value of the chosen_spirit to the amount from the "Total Value" column in the inventory dataframe.  Match the name of the chosen_spirit to the name in the inventory dataframe regardless of case
    df = pd.DataFrame.from_dict(inventory, orient = 'columns')
    total_value = df[df['Name'].str.lower() == st.session_state.chosen_spirit.lower()]['Total Value'].values[0]
    # Note the difference in the value of the chosen_spirit in inventory and the total profit
    st.success(f"Congratulations!  You turned \${total_value:.2f} worth of {st.session_state.chosen_spirit} into ${total_profit:.2f} worth of profit!")

if st.session_state.inventory_page == 'get_inventory_choice':
    get_inventory_choice()
elif st.session_state.inventory_page == 'upload_inventory':
    upload_inventory()
elif st.session_state.inventory_page == 'choose_spirit':
    choose_spirit()
elif st.session_state.inventory_page == 'create_cocktail':
    create_cocktail()
elif st.session_state.inventory_page == 'display_recipe':
    display_recipe()
elif st.session_state.inventory_page == 'display_cost':
    display_cost()

