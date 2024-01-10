import os
from datetime import datetime, timedelta
import instaloader
import openai
from firebase_admin import credentials, initialize_app, storage, firestore

# Initialize Firebase Admin SDK
cred = credentials.Certificate(
    "slugevents-57e0b-firebase-adminsdk-brd1a-dcc2a15087.json")
initialize_app(cred, {
    "storageBucket": "slugevents-57e0b.appspot.com"
})

# Initialize Instaloader
L = instaloader.Instaloader()

# OpenAI API Key
openai.api_key = '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'

# Instagram profiles to download images from
profiles = [
    "ucsc9_jrl",
    "porter.college",
    "kc_ucsc",
    "stevenson.ucsc",
    "cowell.ucsc",
    "rcc_ucsc",
    "oakescollege",
    "ucsccrowncollege",
    "ucscmerrillcollege"
]

# Firebase Storage and Firestore clients
bucket = storage.bucket()
db = firestore.client()

# Create 'instagram_images' directory if it doesn't exist
os.makedirs("instagram_images", exist_ok=True)

# Download and process Instagram images
one_week_ago = datetime.now() - timedelta(weeks=1)
for profile_name in profiles:
    profile = instaloader.Profile.from_username(L.context, profile_name)

    for post in profile.get_posts():
        post_date = post.date_utc.replace(tzinfo=None)  # Remove timezone info

        if post_date < one_week_ago:
            continue  # Skip posts older than a week

        post_id = post.shortcode  # Use the post's shortcode as a unique identifier

        # Check if the post already exists in the Firestore database
        existing_post = db.collection("events").document(post_id).get()
        if existing_post.exists:
            print(f"Skipping duplicate post: {post_id}")
            continue

        # Determine if the post describes an event
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{post.caption}\nDetermine if this instagram post description describes an event. If the description doesn't describe an event, output 'not an event'. If it does, output the location, date, and time of the event in the following format: 'location: location_placeholder, date: date_placeholder, time: time_placeholder"}
            ]
        )

        print(post.caption)
        print(response)

        # If the model determines it's not an event, skip this post
        if 'not an event' in response['choices'][0]['message']['content'].lower():
            continue

        # Extract location, date, and time from the response
        location, date, time = None, None, None  # Initialize variables
        message_parts = response['choices'][0]['message']['content'].split(
            ', ')
        for part in message_parts:
            if 'location:' in part.lower():
                location = part.split(': ')[1]
            elif 'date:' in part.lower():
                date = part.split(': ')[1]
            elif 'time:' in part.lower():
                time = part.split(': ')[1]

        # Check if the post is a sidecar (album)
        if post.typename == "GraphSidecar":
            # Convert generator to list
            sidecar_nodes = list(post.get_sidecar_nodes())
            # Get the URL of the first image
            url = sidecar_nodes[0].display_url
        else:
            url = post.url

        # Download image
        image_name = f"{profile_name}_{post.date_utc.strftime('%Y%m%d_%H%M%S')}"
        image_path = os.path.join("instagram_images", image_name)
        L.download_pic(image_path, url, post.date_utc)

        # Append the file extension to the image_path
        image_path += ".jpg"

        # Upload image to Firebase Storage
        blob = bucket.blob(f"{image_name}.jpg")
        blob.upload_from_filename(image_path)
        blob.make_public()

        # Get image URL from Firebase Storage
        image_url = blob.public_url

        # Store image metadata in Firestore
        doc_ref = db.collection("events").document(post_id)
        doc_ref.set({
            "account": profile_name,
            "description": post.caption,
            "date_posted": post.date_utc,
            "imageSrc": image_url,
            "eventLocation": location,
            "eventDate": date,
            "eventTime": time
        })

        print(f"Processed {image_name}.jpg")
