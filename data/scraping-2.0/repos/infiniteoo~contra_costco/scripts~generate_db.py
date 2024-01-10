from openai import OpenAI
from dotenv import load_dotenv
import os, json
import boto3
import requests
from io import BytesIO



load_dotenv()

# initialize openAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI()

# initialize s3 client
s3 = boto3.resource('s3')
s3_client = boto3.client('s3')
bucket_name = 'contra-costco'


# initialize dynamodb client
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('contra_costco')


def insert_item():
    for _ in range(33):
        system_content = f"You have a great sense of humor who outputs responses in JSON format.  For this and future requests we will be generating a database of funny products for a sarcastic parody store called Contra Costco.  Based on the 80's arcade game Contra, this e-commerce grocery store exclusively sells over-the-top, comical items that a 80's jungle freedom fighter would need.  Some examples: Rapid Fire Banana Launcher, Explosive Energy Drinks, Pixel Power Camoflauge Clothing.  Please try not to make the same items over and over, and don't use same product names.  Avoid using the same words, IE 'Guerilla' multiple times. Be creative and extremely diverse.  For the returned JSON data please always use these specific keys: product_name, product_description, product_price, product_features, product_rating, product_reviews, product_use_cases, product_tags, product_categories.  Please use the following values for the keys: product_name: string, product_description: string, product_price: string, product_features: list, product_rating: string, product_reviews: list, product_use_cases: list, product_tags: list, product_categories: list."

        response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": "What is the product name?"},
                {"role": "user", "content": "What is the product description?"},
                {"role": "user", "content": "What is the product price?"},
                {"role": "user", "content": "What are the product features? Please give three features."},
                {"role": "user", "content": "What is the product rating? Please give a rating between 1 and 5."},
                {"role": "user", "content": "What are the product reviews? Please give three reviews."},
                {"role": "user", "content": "What are the product use cases? Please give three use cases."},
                {"role": "user", "content": "What are the product tags? Please give three tags."},
                {"role": "user", "content": "What are the product categories? Please give three categories."},
            ],
        )
    

        for choice in response.choices:

                try:
                    content = json.loads(choice.message.content)
                except json.JSONDecodeError:
                    print(f"Failed to parse JSON content: {choice.message.content}")
                    continue
                # Assuming 'content' is now a dictionary
                print(f"content: {content}")

        # Save the brand to the database
        print(f"")
        print(f"Saving {content['product_name']} to database...")

       

        # write to dynamodb
        table.put_item(
                Item={
                    'product_name': content['product_name'],
                    'product_description': content['product_description'],
                    'product_price': content['product_price'],
                    'product_features': content['product_features'],
                    'product_rating': content['product_rating'],
                    'product_reviews': content['product_reviews'],
                    'product_use_cases': content['product_use_cases'],
                    'product_tags': content['product_tags'],
                    'product_categories': content['product_categories']
                }
            )


def generate_images():
    # get all items from dynamodb
    response = table.scan()
    items = response['Items']

    # iterate through items
    for item in items:
        try:
            # Construct prompt with item details
            prompt = f"Please create an image of a product for a comical parody store called Contra Costco. Based on the 80's arcade game Contra, this e-commerce grocery store exclusively sells over-the-top, comical items that an 80's jungle freedom fighter would need. Please make the image colorful, friendly, fun, bright, silly, comical, and have a transparent background.\nProduct Name: {item['product_name']}\nProduct Description: {item['product_description']}\nProduct Features: {', '.join(item['product_features'])}\nProduct Reviews: {', '.join(item['product_reviews'])}\nProduct Use Cases: {', '.join(item['product_use_cases'])}\nProduct Tags: {', '.join(item['product_tags'])}\nProduct Categories: {', '.join(item['product_categories'])}"

            print(f"Prompt: {prompt}")

            # Generate image
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )

            # Get image URL
            dall_e_image_url = response.data[0].url

            # Download image
            response = requests.get(dall_e_image_url)
            image = BytesIO(response.content)

            s3_image_key = f"contra-costco/{item['product_name']}.png"
            s3_client.upload_fileobj(image, bucket_name, s3_image_key)

            s3_image_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_image_key}"

            # Update the item in DynamoDB with the image URL
            table.update_item(
                Key={'product_name': item['product_name']},
                UpdateExpression='SET product_image_url = :val1',
                ExpressionAttributeValues={':val1': s3_image_url}
            )

        except Exception as e:
            # Handle any errors here, e.g., log the error message
            print(f"Error processing item {item['product_name']}: {str(e)}")

    # Move the return statement outside the loop if needed
    # return json({"url": s3_image_url})

        


     


        

       

        

if __name__ == "__main__":
    # insert_item()
    generate_images()
    print("All items inserted successfully.")
    



