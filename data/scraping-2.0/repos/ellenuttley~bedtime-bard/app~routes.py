"""
This file contains all our routes, which are accessed via a blueprint which is also defined here.
The individual routes also contain functionality for the forms, and for the story generation.
As well as the functionality for boilerplate functioning, eg. log in and log out.

We used flask-login for the login authentication, and this has inbuilt methods and decorators that are used throughout :
- current_user.is_authenticated : checks whether a user is logged in
- @login_required : states that a user cannot access a certain route unless they are logged in, and if they try it
  routes them to the login page. I hoped to also have this show a flash message, but I couldn't get it to work
"""

from flask import current_app as app
from .models import db, User, BedtimeSteps, UserBedtimeRoutine, CreatureChoice, StoryTypeChoice, DislikeChoice, UserCreature, UserStoryType, UserDislike
from flask_login import LoginManager, login_user, logout_user, login_required, UserMixin, logout_user, current_user
import openai
from flask import render_template, request, redirect, url_for, session, Blueprint, flash
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func
from .forms import NewUserForm, ChooseStoryElements, BedtimeRoutineForm
from .openai import Story, Popup
from . import login_manager
import random

# Our API key :


# The Blueprint for the routes, that is the imported in init.py :
routes_bp = Blueprint('routes', __name__)

# Route for the About Us page (navigated to via the navbar) :
@routes_bp.route("/about_us")
def about():
    return render_template("about.html", page_name='about')

# # Route for the Bookshelf page :       (not currently functional)
# @routes_bp.route('/bookshelf')
# def bookshelf():
#     return render_template('bookshelf.html')

# Route for the Homepage :
@routes_bp.route("/")
def home():
    return render_template("home.html", page_name='homepage')

# Loads current user to the session. Used elsewhere via 'user = current_user' :
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Route for logout functionality :
# (this acts only as a redirect, and doesn't render a template)
@routes_bp.route('/logout', methods=['GET', 'POST'])
def logout():
    if current_user.is_authenticated:
        logout_user()
        session.pop('name', None)            # when they are logged out, their name is removed from the current session
        # messages = get_flashed_messages(category_filter='success')  # not functional
        return redirect(url_for('routes.home'))
    else:
        # messages = get_flashed_messages(category_filter='error')    # not functional
        return redirect(url_for('routes.home'))


# Route for login page :
@routes_bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:                        # if they are already logged in, it routes to the userpage
        return redirect(url_for('routes.user_profile'))
    if request.method == 'POST':                             # form to take the login details
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username, password=password).first()   # queries the database to check them
        if user:                                             # if the check passes ..
            login_user(user)                                 # they are logged in
            session['name'] = current_user.name              # and their name added to the session
            return redirect(url_for('routes.user_profile'))  # they are then routed to the user page
        else:
            # flash("Opps, I don't think you've been here before!", 'error')
            return render_template('signup.html', error=True)
    return render_template('login.html', page_name='login')  # if check fails, they are rerouted back to login

# THE NEXT THREE ROUTES ARE ALL THE SIGNUP PROCESS ---------------------------------------------------------------------
# Registration step : 1 / 3 - enters information into the User model
# Route for the NewUserForm in forms.py
@routes_bp.route('/signup', methods=['GET', 'POST'])
def signup():
    form = NewUserForm()
    if form.validate_on_submit():
        # This adds the inputted information into the appropriate column in the user table :
        user = User(username=form.username.data, password=form.password.data, name=form.name.data.title(), age=form.age.data,
                    pronouns=form.pronouns.data)

        db.session.add(user)
        db.session.commit()
        login_user(user)
        session['name'] = current_user.name
        return redirect(url_for('routes.story_elements'))           # It then redirects to the next form
    else:
        return render_template('signup.html', form=form, page_name='signup')

# Registration step : 2 / 3 - enters information into the User StoryTypes / Dislikes / Creature models
# Route for the StoryElementForm in forms.py
@routes_bp.route('/story_elements', methods=['GET', 'POST'])
def story_elements():
    # Gets current users information :
    user_id = current_user.id
    user = User.query.get(user_id)
    se_form = ChooseStoryElements()       # Defines the form
    creature_choices = [(creature.label, creature.label) for creature in CreatureChoice.query.all()]
    story_type_choices = [(story_type.label, story_type.label) for story_type in StoryTypeChoice.query.all()]
    dislike_choices = [(dislike.label, dislike.label) for dislike in DislikeChoice.query.all()]

    # Populates choices for the form fields
    # Those choices are all the entries in the 'label' column of the Creature/StoryType/DislikeChoices models in models.py
    se_form.creature_choices.choices = creature_choices
    se_form.story_type_choices.choices = story_type_choices
    se_form.dislikes_choices.choices = dislike_choices

    # Handles the form submission :
    if se_form.validate_on_submit():
        # Gets the selected choices :
        selected_creatures = se_form.creature_choices.data
        selected_story_types = se_form.story_type_choices.data
        selected_dislikes = se_form.dislikes_choices.data

        # Uses those choices to create a list of ID numbers to identify each one
        creature_ids = [CreatureChoice.query.filter_by(label=choice).first().id for choice in selected_creatures]
        story_type_ids = [StoryTypeChoice.query.filter_by(label=choice).first().id for choice in selected_story_types]
        dislike_ids = [DislikeChoice.query.filter_by(label=choice).first().id for choice in selected_dislikes]

        # Then enters those id numbers into the UserCreature/StoryType/Dislike, along with the current user id
        for idx, creature_id in enumerate(creature_ids):
                user_creature = UserCreature(user_id=user_id, creature_id=creature_id)
                db.session.add(user_creature)
                db.session.commit()

        for idx, story_type_id in enumerate(story_type_ids):
            user_story_type = UserStoryType(user_id=user_id, story_type_id=story_type_id)
            db.session.add(user_story_type)
            db.session.commit()

        for idx, dislike_id in enumerate(dislike_ids):
            user_dislike = UserDislike(user_id=user_id, dislike_id=dislike_id)
            db.session.add(user_dislike)
            db.session.commit()

        return redirect(url_for('routes.bedtime_steps'))   # then redirects to the next form

    return render_template('story_elements_form.html', se_form=se_form, page_name='signup')

# Registration step : 3 / 3 - enters information into the UserBedtimeRoutine model
# Route for the BedtimeRoutineForm in forms.py
@routes_bp.route('/bedtime_steps', methods=['GET', 'POST'])
def bedtime_steps():
    # Gets current users information :
    user_id = current_user.id
    user = User.query.get(user_id)
    form = BedtimeRoutineForm()

    # Populates choices for the form fields
    # Those choices are all the entries in the 'label' column of the BedtimeSteps model in models.py
    bedtime_step_choices = [step.label for step in BedtimeSteps.query.all()]
    form.bedtime_step_1.choices = bedtime_step_choices
    form.bedtime_step_2.choices = bedtime_step_choices
    form.bedtime_step_3.choices = bedtime_step_choices
    form.bedtime_step_4.choices = bedtime_step_choices
    form.bedtime_step_5.choices = bedtime_step_choices
    if form.validate_on_submit():

        # Handles the form submission :
        # Get the form choices, and store in a list :
        step_choices = [form.bedtime_step_1.data, form.bedtime_step_2.data, form.bedtime_step_3.data,
                        form.bedtime_step_4.data, form.bedtime_step_5.data]

        # Use those choices to get the corresponding id numbers from the BedtimeSteps model in models.py:
        step_id = [BedtimeSteps.query.filter_by(label=choice).first().id for choice in step_choices]

        # Then adds that id number, the user_id, and a number 1 - 5 to state what step the choice was for
        step_1 = UserBedtimeRoutine(user_id=user.id, bedtime_step_id=step_id[0], position=1)
        db.session.add(step_1)

        step_2 = UserBedtimeRoutine(user_id=user.id, bedtime_step_id=step_id[1], position=2)
        db.session.add(step_2)

        step_3 = UserBedtimeRoutine(user_id=user.id, bedtime_step_id=step_id[2], position=3)
        db.session.add(step_3)

        step_4 = UserBedtimeRoutine(user_id=user.id, bedtime_step_id=step_id[3], position=4)
        db.session.add(step_4)

        step_5 = UserBedtimeRoutine(user_id=user.id, bedtime_step_id=step_id[4], position=5)
        db.session.add(step_5)

        db.session.commit()
        login_user(user)                                     # at the end of the process it then logs the user in
        session['name'] = current_user.name                  # adds their name to the session

        return redirect( url_for('routes.user_profile'))     # and routes them to the user page, to view their story

    else:
        return render_template('bedtime_steps.html', form=form, page_name='signup')

# END OF SIGNUP PROCESS ------------------------------------------------------------------------------------------------

# Route for the Userpage :
# This page contains the button that activates the story generation
@routes_bp.route('/user', methods=['GET', 'POST'])
@login_required
def user_profile():
    return render_template('userpage.html', page_name='userpage')

# Route for the story generation process - does not render a template
# Until this route finishes and redirects, the button in the userpage becomes a loading button
@routes_bp.route("/story_generation")
@login_required
def generate_story():
    # Gets current users information :
    user_id = current_user.id
    user = User.query.get(user_id)

    # STORY GENERATION CODE : -----------------------------------------------------------------------------------------

    # STEP 1 - GET THE STORY DATA
    # All methods used below are contained in the User model in models.py
    # user data for story :
    name = current_user.get_name()
    age = current_user.get_age()
    pronouns = current_user.get_pronouns()
    story_type = current_user.get_story_type()
    creature = current_user.get_creature()
    dislikes = current_user.get_dislikes()

    # bedtime steps data for popups :
    routine_steps = current_user.get_routine()

    # STEP 2 - USE DATA TO DEFINE STORY OBJECT :
    # Story class defined in openai.py
    current_story = Story(name, age, pronouns, story_type, creature, dislikes)

    # STEP 3 - GENERATE STORY : ------------------------------------------------
    # The generate_story method is contained in the Story class in openai.py

    # The story is generated, and returned as a list of individual content pieces, which is then added to the session
    # These pieces are used to inform the popup prompts outlined below
    # The content in list is then shown in sequence in the route : show_story (below)
    story_parts = current_story.generate_story()
    session['story_parts'] = story_parts

    # STEP 4 - DEFINE POPUP OBJECTS : -----------------------------------------
    # Popup class defined in openai.py
    # Uses the name variable and the bedtime routine retrieved in step 1, and the story_parts from step 3
    popup_1 = Popup(name, routine_steps[0], story_parts[0])
    popup_2 = Popup(name, routine_steps[1], story_parts[1])
    popup_3 = Popup(name, routine_steps[2], story_parts[2])
    popup_4 = Popup(name, routine_steps[3], story_parts[3])
    popup_5 = Popup(name, routine_steps[4], story_parts[4])

    # STEP 5 - GENERATE POPUP TEXT : ------------------------------------------
    # The generate_popups method is contained in the Popup class in openai.py

    # Adds popups defined in step 4 to a list (I thought this was the most readable way) :
    pop_ups = [popup_1, popup_2, popup_3, popup_4, popup_5]

    # .. then uses that list of popup objects to generate the content for the individual pop_ups.
    # These are then saved to a list, and the list is added to the session.
    # The content of this list is then shown in sequence in the route : show_popup (below)
    bedtime_routine = Popup.generate_popups(pop_ups)
    session['bedtime_routine'] = bedtime_routine

    # When the story generation above is complete - the user is rerouted to the first page of the story process
    # .. this then start the story and popup loop.
    return redirect( url_for('routes.show_story', num=0))

    # END OF STORY GENERATION CODE -----------------------------------------------------------------------------------

# The two routes below are for the story and popup loop.
# It starts at show_story/0, show_popup/0, then logic in show_popup route below then increments the number by 1
# .. Until it gets to 5, at which time end_of_story.html is routed to.
# (The users then have the options to start this process again, or return to the homepage)

# To see the story progression in order, start from the url: http://127.0.0.1:5000/user and click on the book :)
# please sign up / in first

# Route for the story pieces :
@routes_bp.route("/show_story/<int:num>")
@login_required
def show_story(num):
    if num == 5:
        session.pop('story_parts', None)              # removes the current story from session
        session.pop('bedtime_routine', None)          # removes bedtime_routine from session
        return render_template('end_of_story.html', modal=True)
    return render_template('basic_story.html', story_parts=session.get('story_parts'), num=num, page_name='story')

# Route for the popups :
@routes_bp.route("/show_popup/<int:num>")
@login_required
def show_popup(num):
    num +=1
    return render_template('basic_popup.html', bedtime_routine=session.get('bedtime_routine'), num=num, page_name='story', modal=True)


