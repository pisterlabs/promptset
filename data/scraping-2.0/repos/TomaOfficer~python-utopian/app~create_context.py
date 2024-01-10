import os
import openai
from flask import Blueprint, request, session, redirect, url_for, jsonify, render_template
from app.models import Restaurant, User
from app.extensions import db
from flask_login import current_user, login_required
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

# Load environment variables and initialize OpenAI client
openai.api_key = os.getenv('OPENAI_API_KEY')

context_blueprint = Blueprint('context', __name__)

@context_blueprint.route('/create_context', methods=['POST'])
@login_required
def create_context():
    user_input = request.form['user_input']

    try:
        prompt = f"Before you can answer the following question, what context do you need to know about the user? Here is the user's question: {user_input}"
        
        response = openai.chat.completions.create(
            model="gpt-4-1106-preview",
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a helpful assistant designed to identify and output the types of context needed to answer a user's question. Context types might include 'Income', 'Dental Health', 'Medical History', etc."},
                {"role": "user", "content": prompt},
            ]
        )

        # Parsing the response to extract context requirements
        context_requirements = response.choices[0].message.content
        formatted_response = {"response": context_requirements}  # Define formatted_response

        return jsonify(formatted_response)

    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500

class RestaurantForm(FlaskForm):
    legal_name = StringField('Legal Name', validators=[DataRequired()])
    business_structure = StringField('Business Structure', validators=[DataRequired()])
    ein = StringField('Employer Identification Number (EIN)', validators=[DataRequired()])
    business_address = StringField('Business Address', validators=[DataRequired()])
    business_nature = StringField('Nature of Business', validators=[DataRequired()])
    owner_info = StringField('Owner Information', validators=[DataRequired()])
    governing_law = StringField('Governing Law', validators=[DataRequired()])
    contact_details = StringField('Contact Details', validators=[DataRequired()])
    submit = SubmitField('Submit')

@context_blueprint.route('/create_restaurant', methods=['GET', 'POST'])
@login_required
def add_restaurant():
    form = RestaurantForm()
    if form.validate_on_submit():
        restaurant = Restaurant(
            legal_name=form.legal_name.data,
            business_structure=form.business_structure.data,
            ein=form.ein.data,
            business_address=form.business_address.data,
            business_nature=form.business_nature.data,
            owner_info=form.owner_info.data,
            governing_law=form.governing_law.data,
            contact_details=form.contact_details.data,
            user_id=current_user.id  # Assuming Flask-Login is used for user management
        )
        # Add other fields as necessary
        db.session.add(restaurant)
        db.session.commit()
        success_message = "Restaurant added successfully!"
    
    return render_template('create-context.html', form=form, message=success_message)

@context_blueprint.route('/use_restaurant', methods=['POST'])
@login_required
def use_restaurant():
    restaurant_id = request.form.get('restaurant_id')
    restaurant = Restaurant.query.get(restaurant_id)
    if restaurant and restaurant.user_id == current_user.id:
        prompt = f"I'm managing a restaurant named {restaurant.legal_name}, " \
                 f"which is a {restaurant.business_structure} located at {restaurant.business_address}. " \
                 f"It has the following nature of business: {restaurant.business_nature}. " \
                 f"Can you provide some advice on how to improve customer satisfaction?"

        response = openai.chat.completions.create(
            model="gpt-4-1106-preview",
            temperature=0.1,
            messages=[
                {"role": "system", "content": "You are a knowledgeable assistant providing business advice."},
                {"role": "user", "content": prompt},
            ]
        )
        chat_response = response.choices[0].message.content if response.choices else "No response received."
        return jsonify({'response': chat_response})

    else:
        return jsonify({'error': 'Restaurant not found or access denied'}), 404
