import openai
import requests
import os
from PIL import Image, ImageDraw, ImageFont
import datetime
import email_config

# Set your API key
openai.api_key = email_config.OPENAI_API_KEY

def send_email(subject, recipient, body, img_path=None):
    import email_config
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    from email.mime.image import MIMEImage
    import os
    

    EMAIL_ADDRESS = email_config.EMAIL_ADDRESS_AUTO
    EMAIL_PASSWORD = email_config.EMAIL_PASSWORD_AUTO

    # Create the email message
    message = MIMEMultipart()
    message['From'] = EMAIL_ADDRESS
    message['To'] = recipient
    message['Subject'] = subject

    message.attach(MIMEText(body, 'html'))

    # If an image path is provided, add the image as an inline attachment
    if img_path is not None:
        with open(img_path, 'rb') as img_file:
            img_data = img_file.read()
        img_mime = MIMEImage(img_data)
        img_mime.add_header('Content-ID', '<{}>'.format(os.path.basename(img_path)))
        img_mime.add_header('Content-Disposition', 'inline', filename=os.path.basename(img_path))
        message.attach(img_mime)


    # Connect to the Gmail SMTP server and send the email
    with smtplib.SMTP('smtp.gmail.com', 587) as smtp:
        smtp.starttls()
        smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        smtp.send_message(message)

def call_gpt_images(content, email_address):
    response = openai.Image.create(
    model="dall-e-3",
    prompt=content,
    n=1,
    size="1024x1024"
    )

    # Create a directory to store the images if it doesn't exist
    if not os.path.exists('gpt_images'):
        os.makedirs('gpt_images')

    # Extract the URL
    image_description = response["data"][0]["revised_prompt"]
    image_url = response["data"][0]["url"]

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H-%M-%S")

    response = requests.get(image_url)
    if response.status_code == 200:
        image_filename = f'image_{timestamp}.png'  # Replace colon with underscore
        image_path = os.path.join('gpt_images', image_filename)
        with open(image_path, 'wb') as f:
            f.write(response.content)

    # Load the image
    image = Image.open(image_path)

    # Prepare the watermark text
    watermark_text = "By Lazaro Gonzalez"

    # Create a drawing context
    draw = ImageDraw.Draw(image)

    # Specify the font and size of the watermark
    font = ImageFont.truetype('arial.ttf', 15)  # Adjust the font and size as needed

    # Get the bounding box for the watermark text
    textbbox = draw.textbbox((0, 0), watermark_text, font=font)

    # Position for the watermark (center of the image)
    width, height = image.size
    x = (width - textbbox[2]) / 2
    y = (height - textbbox[3]) / 2

    # Add the watermark text
    draw.text((x, y), watermark_text, font=font, fill=(255, 255, 255))

    

    # Save the watermarked image with a different filename
    watermarked_image_filename = f'watermarked_image_{timestamp}.png'  # Replace colon with underscore
    watermarked_image_path = os.path.join('gpt_images', watermarked_image_filename)
    watermarked_image_path = os.path.abspath(watermarked_image_path)
    watermarked_image_path = r'{}'.format(watermarked_image_path)
    watermarked_image_path = 'gpt_images.png'
    
    image.save(watermarked_image_path)

    # prepare email
    subject = "GPT Image"
    recipient = email_address
    body = f"""
    <html>
        <body>
            <h1>DALL-E Generated Image</h1>
            <p>{image_description}</p>
            <p><img src="cid:{os.path.basename(watermarked_image_path)}" alt="Generated Image"></p>
        </body>
    </html>
    """
                                                                                                                                                               
    send_email(subject, recipient, body, watermarked_image_path)
