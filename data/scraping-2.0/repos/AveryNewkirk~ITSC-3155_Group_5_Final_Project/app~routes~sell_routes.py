from collections import UserDict
from fileinput import filename
from typing import Iterable
from flask import Blueprint, abort, current_app, jsonify, render_template, request, redirect, url_for, session
import os
from ..models.pipeline import db, Users, Listing, Album, Photo
from ..utils import upload_file
from openai import OpenAI
import base64
import requests
from werkzeug.utils import secure_filename
from urllib.parse import urlparse, unquote

# Constants and variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
aiClient = OpenAI()

# Create Flask Blueprint
sell = Blueprint('sell', __name__)

@sell.route('/sell', methods=["GET", "POST"])
def create_listing():
    if not session['username']:
        return redirect('/login')
    username = session['username']
    
    user = Users.query.filter_by(username=username).first()
    
    if not isinstance(user, Users):
        abort(401)
    user_id = user.user_id
    
    if request.method == 'POST':
        # Extract form fields
        title = request.form['listing-title']
        description = request.form['item-description']
        price = float(request.form['price'])
        album_id = int(request.form['album-id'])
        listing_photos = request.files.getlist('upload-pictures')
        
        # If no Album already exists for the listing, create Album object for images
        if not album_id or not isinstance(album_id, int):
            listing_album = Album(user_id)
            db.session.add(listing_album)
            db.session.commit()
            album_id = listing_album.album_id
        
        # Create new listing
        new_listing = Listing(title=title, description=description, price=price, user_id=user_id, album_id=album_id)
        db.session.add(new_listing)
        db.session.commit()
        
        # Get model images and add to database
        model_images = []
        for i in range(1, 5):
            image_field = f'model-image-url-{i}'
            if image_field in request.form and request.form[image_field]:
                model_images.append(request.form[image_field])
                
        # Download and save images
        model_image_urls = []
        for url in model_images:
            response = requests.get(url)
            if response.status_code == 200:
                filename = url_to_filename(url)
                filepath = os.path.join(current_app.root_path, 'static', 'user_images', filename)
                with open(filepath, 'wb') as file:
                    file.write(response.content)
                model_image_urls.append(filename)
                
        # Create Photo objects and add to db
        for image_url in model_image_urls:
            if image_url:
                photo = Photo(album_id=album_id, photo_url=image_url)
                db.session.add(photo)
        db.session.commit()
        
        return redirect(url_for('sell.sell_success', listing_id=new_listing.listing_id))
        
    # Default to showing empty Sell page
    return render_template('sell.html')

@sell.route('/upload_images', methods=['POST'])
def upload_images():
    if not session['username']:
        return redirect('/login')
    username = session['username']
    
    user = Users.query.filter_by(username=username).first()
    
    if not isinstance(user, Users):
        abort(401)
    user_id = user.user_id
        
    photos = request.files.getlist('upload-pictures')
    
    # Create Album object for images
    album = Album(user_id)
    db.session.add(album)
    db.session.commit()
    
    # Create Photo objects and add to db
    for file in photos:
        if file and file.filename:
            photo_url = upload_file(file)
            if photo_url:
                photo = Photo(album_id=album.album_id, photo_url=photo_url)
                db.session.add(photo)
    
    db.session.commit()
    
    # Get album_id
    return {'album_id': album.album_id}

@sell.route('/generate_description',  methods=['POST'])
def handle_generate_description():
    if not session['username']:
        return redirect('/login')
    try:
        data = request.get_json()
        album_id = data.get('album_id')
        response = generate_description(album_id)
        description = response.choices[0].message.content if response else ''
        return jsonify(description=description)
    except Exception as e:
        return jsonify(error=str(e)), 500
    
@sell.route('/generate_pictures',  methods=['POST'])
def generate_pictures():
    if not session['username']:
        return redirect('/login')
    
    data = request.get_json()
    description = data.get('description')
    
    concise_prompt = '''
    You will be provided with the description of a clothing item from an online store. 
    Please abstract out the physical attributes of the clothing item, 
    including type, fit, style, material, color, pattern/design, and detailing. 
    Please return a concise summary of these physical attributes without any additional filler or fluff. \n
    Clothing item description: \n
    '''
    
    if description:
        concise_prompt += description
    else:
        return jsonify({'error': 'Description is missing'}), 400
    
    response_1 = aiClient.chat.completions.create(
        model="gpt-4",
        messages=[{
            "role": "system",
            "content": concise_prompt
        }]
    )
    
    concise_description = response_1.choices[0].message.content
    
    image_prompt = '''
    You will be provided with a detailed summary of the physical attributes of a clothing item.
    Using these attributes, create a true-to-life photograph of the clothing item on an appropriate human model. 
    The model is posing in full view against a white backdrop. 
    Clothing item features ALL attributes from the clothing item physical attributes summary. \n
    Here is the clothing item physical attributes summary: \n
    '''
    
    if concise_description:
        image_prompt += concise_description
    else:
        return jsonify({'error': 'Concise description is missing'}), 400
    
    response_2 = aiClient.images.generate(
        model="dall-e-3",
        prompt=image_prompt,
        size="1024x1024",
        quality="standard",
        style='vivid',
        n=1
    )
    
    image_url = response_2.data[0].url
    revised_prompt = response_2.data[0].revised_prompt
    return jsonify({'image_url': image_url})

@sell.route('/sell_success/<int:listing_id>')
def sell_success(listing_id):
    if not session['username']:
        return redirect('/login')
    username = session['username']
    
    listing = Listing.query.get(listing_id)
    if not isinstance(listing, Listing):
        abort(401)
    
    if not username == listing.user.username:
        abort(401)
    return render_template('sell_success.html', listing=listing)

# UTILITY FUNCTIONS
# Function to encode images in base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
# Function to get mime type
def get_mime_type(file_path):
    mime_types = {
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'png': 'image/png',
        'tiff': 'image/tiff',
        'webp': 'image/webp'
    }
    ext = file_path.split('.')[-1].lower()
    return mime_types.get(ext, 'image/jpeg')

# Function to generate AI generated model pictures
def generate_description(album_id):
    # List to hold message content to OpenAI API
    content = []
    # Initial prompt
    content.append({
        "type": "text", 
        "text": '''
        The attached image is a photograph of a single item of clothing. 
        Any additional attached images are additional photographs of the same item of clothing taken from various angles.

        Your task is to create a highly detailed description of the clothing item shown in the photograph(s) that is no more than 180 words in length. 
        The description should be extremely thorough and include any and all possible information about the clothing item, while also concise enough to be no longer than 180 words. 
        At the very least, the description should describe each of the following attributes of the clothing item in comprehensive detail:

        List of clothing item attributes:
        1. Type: Defines the category of clothing, such as a shirt, pants, dress, jacket, etc.
        2. Style: The specific design or fashion style, such as casual, formal, vintage, modern, etc.
        3. Color: The primary and secondary colors, including shades and tones.
        4. Material: The material used, like cotton, polyester, silk, leather, etc., including blends.
        5. Texture: The feel of the material, like smooth, rough, ribbed, plush, etc.
        6. Fit: How the clothing fits the body, like loose, slim, tailored, oversized, etc.
        7. Pattern/Design: Any prints or patterns on the fabric, like stripes, floral, polka dots, abstract, etc.
        8. Detailing: Any features like embroidery, lace, buttons, zippers, pockets, ruffles, etc.

        Take a minimal approach to sentence structure and cut out unnecessary words to stay within the 180-word length limit. 
        Avoid simply listing out information and write in full sentences instead. 
        As you write, act as though the clothing item is right in front of you and do not acknowledge the existence of the image whatsoever. 
        Finally, write the description in the style of an online seller writing a description for a listing.
        '''
    })
    
    # Adding images to content
    album = Album.query.get(album_id)
    if isinstance(album, Album):
        photos = album.photos
        if isinstance(photos, Iterable):
            for photo in photos:
                image_path = os.path.join(current_app.root_path, 'static', 'user_images', photo.photo_url)
                base64_image = encode_image(image_path)
                mime_type = get_mime_type(image_path)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}"
                    }
                })
    
    # Sending request to OpenAI gpt-4-vision
    response = aiClient.chat.completions.create(
        model = "gpt-4-vision-preview",
        messages = [{
            "role": "user",
            "content": content,
        }],
        max_tokens = 300
    )
    
    return response

def url_to_filename(url: str):
    parsed_url = urlparse(url)
    path = parsed_url.path
    filename = path.split('/')[-1]
    return secure_filename(unquote(filename))
