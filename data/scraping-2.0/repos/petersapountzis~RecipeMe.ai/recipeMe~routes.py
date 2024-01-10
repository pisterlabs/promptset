from flask import request, render_template, session, url_for, redirect, flash, request, jsonify, make_response
from recipeMe import app, db, bcrypt, celery
from recipeMe.models import User, Recipe
from recipeMe.login import RegistrationForm, LoginForm
from openai import OpenAI
from . import celery


import re
from flask_login import login_user, current_user, logout_user, login_required
import json
import pdfkit
import os
import time

# path_wkhtmltopdf = os.environ.get('WKHTMLTOPDF_BINARY', '/usr/local/bin/wkhtmltopdf')
# config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)

# home page route
@app.route("/home")
@app.route("/")
def home():
    return render_template('/home.html')

# recipe page route 
@app.route("/register", methods =["GET", "POST"])
def register():
    if current_user.is_authenticated:
        # if logged in send them to home page
        return redirect(url_for('home'))
    form = RegistrationForm()
    if form.validate_on_submit():
        # encrypt password
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode('utf-8')
        user = User(username = form.username.data, email = form.email.data.lower(), password = hashed_password)
        # add user to database
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You are now able to login.', 'success')
        # once registered, make them login
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)


# login page route
@app.route("/login", methods=['GET', 'POST'])
def login():
    # fetch cross site request forgery token
    token = request.form.get('csrf_token')
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data.lower()).first()
        # ensure user exists and password is correct
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            next_page= request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('home'))
        else:
            flash('Login Unsuccessful. Please check username and password', 'danger')
            print('login unsuccessful')
    return render_template('login.html', title='Login', form=form)


@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route("/library")
@login_required
def library():
    recipes = Recipe.query.filter_by(user_id=current_user.id).all()  # get all recipes by the current user
    for recipe in recipes:
        # Convert the ingredients and directions string back to list
        recipe.ingredients = json.loads(recipe.ingredients)
        recipe.directions = json.loads(recipe.directions)
    return render_template('library.html', recipes=recipes)  # render the library page with the recipes



@app.route('/form', methods=["GET", "POST"])
def getFormData():
    if request.method == 'POST':
        # Form has been submitted, store the data in session variables
        session['protein'] = request.form.get("protein", 'any')
        session['cals'] = request.form.get("calories", 'any')
        session['ingredients'] = request.form.get("ingredients", 'any')
        session['servings'] = request.form.get("servings", 1)
        session['cuisine'] = request.form.get("cuisine", 'any')
        session['dish'] = request.form.get("dish", 'any')
        session['allergies'] = request.form.get("allergies", 'any')

        # Start the Celery task for recipe generation
        task = generate_recipe_task.delay(
            session['protein'],
            session['cals'],
            session['ingredients'],
            session['servings'],
            session['cuisine'],
            session['dish'],
            session['allergies']
        )

        # Redirect to the waiting screen, passing the task ID
        # redirect_url = url_for('waiting_screen', task_id=task.id)
        # return jsonify({"redirect_url": redirect_url})
        return redirect(url_for('getGPTResponse'))


    # For a GET request, render and return the recipe request form
    return render_template('index.html')


# Helper function to extract the recipe name, ingredients, directions, and nutrition facts from the GPT response
def extract_recipe_info(recipe_string):
    name_pattern = r"##Name##\n(.*?)\n\n##Ingredients##" or r"##\n(.*?)\n\n##Directions##"
    ingredients_pattern = r"##Ingredients##\n(.*?)\n\n##Directions##"
    directions_pattern = r"##Directions##\n(.*?)\n\n##Nutrition Facts##"
    nutrition_facts_pattern = r"##Nutrition Facts##\n(.*?)\n\n"

    name_match = re.search(name_pattern, recipe_string, re.DOTALL)
    ingredients_match = re.search(ingredients_pattern, recipe_string, re.DOTALL)
    directions_match = re.search(directions_pattern, recipe_string, re.DOTALL)
    nutrition_facts_match = re.search(nutrition_facts_pattern, recipe_string, re.DOTALL)

    recipe_name = name_match.group(1).strip() if name_match else None
    ingredients = ingredients_match.group(1).strip() if ingredients_match else None
    directions = directions_match.group(1).strip() if directions_match else None
    nutrition_facts = nutrition_facts_match.group(1).strip() if nutrition_facts_match else None

    return recipe_name, ingredients, directions, nutrition_facts


# # Helper function to convert the ingredients string to a list
# def ingredients_to_list(ingredients):
#     if ingredients is None:
#         return []
#     # Split the string by commas and strip whitespaces
#     ingredients_list = [ingredient.strip() for ingredient in ingredients.split('-')]
#     ingredients_list = ingredients_list[1:]
#     return ingredients_list


# # Helper function to parse the directions string into a list of instructions
# def parse_instructions(instructions):
#     if instructions is None:
#         return []
#     # Split by digit-period-space pattern, keep the digit and period with the instruction
#     return re.split('\s(?=\d+\.)', instructions)

def ingredients_to_list(ingredients_str):
    if not ingredients_str or "No Ingredients Found" in ingredients_str:
        return []
    # Adjusted to split by newline as each ingredient is on a new line
    return [ingredient.strip() for ingredient in ingredients_str.split('\n') if ingredient.strip()]
def parse_instructions(directions_str):
    if not directions_str or "No Directions Found" in directions_str:
        return []
    # Assuming each step is in a new line
    return [step.strip() for step in directions_str.split('\n') if step.strip()]


protein, cals, ingredients, servings, cuisine, dish = '', '', '', '', '', ''

@celery.task
def generate_recipe_task(protein, cals, dish, ingredients, servings, cuisine, allergies):
    OPENAI_KEY = app.config.get('API_KEY_OPENAI')
    client = OpenAI(api_key=OPENAI_KEY)  # this is also the default, it can be omitted

    print('test hello')
    print('hello ')
    print('api call starting')
    print(OPENAI_KEY)

    system_message= "You are a meal generator programmed to create recipes. Generate a recipe following this structure: Start with the recipe name enclosed within ##Name## tags. List ingredients within ##Ingredients## tags, each ingredient separated by a newline. Provide directions within ##Directions## tags, each step on a new line and prefixed with a number. Conclude with nutrition facts within ##Nutrition Facts## tags. If the user does not provide specific details like ingredients or dish type, use common ingredients or suggest a popular dish. Ensure the response adheres to these formatting rules for easy parsing."
    prompt = f"I want {servings} servings of {cuisine} {dish}, with around {protein} grams of protein and {cals} calories. Please include {ingredients} and exclude any allergens like {allergies}."
    full_prompt = system_message + '\n' + prompt
    try:
        completion = client.chat.completions.create( 
            messages=
            [{
            "role": "system",
            "content": system_message
        },
        {
            "role": "user",
            "content": prompt
        }],
        model="gpt-3.5-turbo",
            temperature=0.5
        )
        print('API call made')
        cleaned_response = completion.choices[0].message.content.strip()
        # Process the response_text as needed
    except Exception as e:
        print("Error in OpenAI API call:", e)
        raise
    name, GPTingredients, directions, nutrition_facts = extract_recipe_info(cleaned_response)

    # DALLE prompt
    image_prompt = f'Generate a high-resolution image of a freshly prepared dish. The dish is a {name}. It is plated on a white, ceramic dish, placed on a rustic wooden table. The lighting should highlight the textures and colors of the dish, making it look appetizing and ready to eat.  In the background, slightly out of focus, there should be a bottle of red wine and a lit candle to create a warm, cozy atmosphere.'

    image = client.images.generate(prompt=image_prompt,
    n=1,
    size="1024x1024")
    image_url = image.data[0].url

    recipe_data = {
        'name': name,
        'ingredients': ingredients_to_list(GPTingredients),
        'directions': parse_instructions(directions),
        'nutrition_facts': nutrition_facts,
        'image_url': image_url
    }

    return recipe_data



@app.route('/recipe', methods=["GET", "POST"])
def getGPTResponse():
    if request.method == 'POST':
        # Get form data
        protein = session.get('protein', 'any')
        cals = session.get('calories', 'any')
        dish = session.get('dish', 'any')
        ingredients = session.get('ingredients', 'any')
        servings = session.get('servings', 1)
        cuisine = session.get('cuisine', 'any')
        allergies = session.get('allergies', 'any')

        # Start the Celery task
        print('starting celery task')
        task = generate_recipe_task.delay(protein, cals, dish, ingredients, servings, cuisine, allergies)

        # Redirect to a waiting screen
        print('task id: ', task.id)
        redirect_url = url_for('waiting_screen', task_id=task.id)
        return jsonify({"redirect_url": redirect_url})
    
    return render_template('index.html')


@app.route('/waiting/<task_id>')
def waiting_screen(task_id):
    return render_template('waiting_screen.html', task_id=task_id)

@app.route('/status/<task_id>')
def task_status(task_id):
    task = generate_recipe_task.AsyncResult(task_id)

    if task.state == 'PENDING':
        return jsonify({'state': task.state, 'status': 'Pending...'})
    elif task.state != 'FAILURE':
        if task.state == 'SUCCESS':
            # Render the recipe page with the task result
            return jsonify({'state': task.state, 'status': 'SUCCESS', 'result': task.result})
        return jsonify({'state': task.state, 'status': 'In Progress...'})
    else:
        return jsonify({'state': task.state, 'status': 'Failed', 'error': str(task.info)})



@app.route('/recipe/show/<task_id>')
def show_recipe(task_id):
    task = generate_recipe_task.AsyncResult(task_id)
    if task.state == 'SUCCESS':
        # Assuming the task result is the recipe data
        return render_template('recipe.html', **task.result)
    # Handle other states or errors as needed


@app.route('/add_to_library', methods=['POST'])
@login_required  # Ensure that a user is logged in before they can add a recipe to the library
def add_to_library():
    # Get the recipe details from the session data
    print('add to library clicked')
    recipe_data = session.get('recipe')
    if recipe_data is None:
        flash('No recipe to add to the library. Please generate a recipe first.', 'warning')
        print('recipe data DNE')
        return redirect(url_for('register'))
        
    # Create a new Recipe and save it to the database
    recipe = Recipe(
        name=recipe_data['name'], 
        ingredients=json.dumps(recipe_data['ingredients']),  # Convert list to string
        directions=json.dumps(recipe_data['directions']),  # Convert list to string
        nutrition_facts=recipe_data['nutrition_facts'], 
        user_id=current_user.id,
        image_url=recipe_data['image_url'])
    
    
    # add recipe to DB if clicked
    db.session.add(recipe)
    db.session.commit()

    # remove the recipe from the session data now
    session.pop('recipe')

    # Redirect the user back to the library or wherever you want
    return redirect(url_for('library'))



@app.route("/delete_recipe/<int:recipe_id>", methods=['POST'])
@login_required
def delete_recipe(recipe_id):
    # Fetch the recipe by id
    recipeToRemove = Recipe.query.get_or_404(recipe_id)

    # Ensure that the current user is the owner of the recipe
    if recipeToRemove.user_id != current_user.id:
        flash('You do not have permission to delete this recipe.', 'error')
        return redirect(url_for('library'))

    # If the current user is the owner of the recipe, delete the recipe
    db.session.delete(recipeToRemove)
    db.session.commit()
    
    flash('Recipe deleted.', 'success')
    return redirect(url_for('library'))


@app.route('/export_recipe/<int:recipe_id>', methods=['POST', 'GET'])
def export_recipe(recipe_id):
    path_wkhtmltopdf = os.getenv('WKHTMLTOPDF_PATH', '/usr/local/bin/wkhtmltopdf')
    config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)

    # Fetch the recipe by id
    recipe = Recipe.query.get_or_404(recipe_id)
    # Convert the ingredients and directions string back to list
    export_ing = json.loads(recipe.ingredients)
    export_dir = json.loads(recipe.directions)
    # nutrition = json.loads(recipe.nutrition_facts)
    # print(nutrition)
    html = render_template('recipe_export.html', ingredients=export_ing, name=recipe.name, directions=export_dir, nutrition_facts=recipe.nutrition_facts, image_url=recipe.image_url)

    # Create a PDF from the HTML
    pdf = pdfkit.from_string(html, False, configuration=config)

    # Create response with the PDF data
    response = make_response(pdf)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'inline; filename={recipe.name}.pdf'
    return response


@app.route('/regenerate', methods=['POST'])
def regenerate():
    # Re-generate a new recipe using GPT
    return redirect(url_for('getGPTResponse'))

