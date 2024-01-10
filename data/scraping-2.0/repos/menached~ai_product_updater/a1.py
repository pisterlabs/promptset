import openai, subprocess, sys, json, html, re, ssl, os, math, glob, pprint, nltk, pdb, requests, time, random  
from PIL import Image, ImageDraw, ImageFont
if not nltk.data.find('tokenizers/punkt'):
    nltk.download('punkt', quiet=True)

sitelist = [
  { "subdomain": "alamo", "site_id": 29 },
  { "subdomain": "burlingame", "site_id": 30 },
  { "subdomain": "campbell", "site_id": 7 },
  { "subdomain": "castrovalley", "site_id": 25 },
  { "subdomain": "concord", "site_id": 31 },
  { "subdomain": "danville", "site_id": 9 },
  { "subdomain": "dublin", "site_id": 8 },
  { "subdomain": "hillsborough", "site_id": 12 },
  { "subdomain": "lafayette", "site_id": 13 },
  { "subdomain": "livermore", "site_id": 14 },
  { "subdomain": "orinda", "site_id": 34 },
  { "subdomain": "pittsburg", "site_id": 28 },
  { "subdomain": "pleasanthill", "site_id": 35 },
  { "subdomain": "sanramon", "site_id": 33 },
  { "subdomain": "walnutcreek", "site_id": 32 }
]

location, sku, startfrom = sys.argv[1], sys.argv[2], int(sys.argv[3])

new_short_description = "If you are searching for exceptional locally sourced cannabis look no further than [showcity] Doap! We have greenhouse flower that is cultivated with meticulous care right here in [showcity]!   The buds are not only kick-ass strong but also bursting with incredible aromas.  We offer a product that is simply Doap!  Get 1hr service by ordering here on the website or calling or SMS [showprettyphone] or WhatsApp at 833-BUY-DOAP(833-289-3627)"
new_description = '''Are you on the lookout for top-notch cannabis flower? Look no further! Our locally sourced cannabis flower, cultivated in the heart of [showcity], is the ultimate choice for those seeking high-quality products. Grown in our carefully controlled greenhouse environment, each strain is nurtured to perfection. From relaxing indicas to energizing sativas, we offer a wide range of strains to suit every preference.\n\nWhat sets our cannabis flower apart is our commitment to natural cultivation practices. Bathed in the warm glow of organic sunlight, our plants thrive without the use of any pesticides or harmful chemicals. This ensures a safe and healthy option for your enjoyment.\n\nWhen it comes to the experience, our buds don't disappoint. Our dense, fragrant flowers boast a superb taste and produce a smooth and flavorful smoke. It's a sensory delight that guarantees a pleasant and satisfying experience.\n\nOrdering our exceptional cannabis flower is a breeze. Simply give our friendly live budtender Steve a call at [showprettyphone]. Steve is here to assist you in selecting and ordering your desired strains. With his expertise, you can rest assured that you'll find the perfect match for your needs.\n\nDon't miss out on the opportunity to try our outstanding greenhouse cannabis flower. Give it a go today and discover the difference it can make in your cannabis experience.'''

credentials = {}
def get_site_id(subdomain):
  for site in sitelist:
    if site["subdomain"] == subdomain:
      return site["site_id"]
  return None
print("Fetch ",location.capitalize(),"product",sku," Push starting at #",startfrom)
creds_file_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),  # Get the directory of the current file
    "../creds2.txt"  # Append the relative path to the credentials file
)
class Location:
    def __init__(self, website, user, city, phone, consumer_key, consumer_secret, api_key):
        self.website = website
        self.user = user
        self.city = city
        self.phone = phone
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.api_key = api_key  # Here's the new attribute

def scp_file_to_remote(local_file, remote_file):
    try:
        # Run SCP command
        subprocess.Popen(["scp", local_file, remote_file])
        print("File transfer initiated.")
        
    except Exception as e:
        print("Error while copying the file:", e)

def download_image(url, filename):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(filename, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Image downloaded successfully: {filename}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {str(e)}")

def add_watermark_and_save(image_path, watermark_text, output_path):
    try:    
        image = Image.open(image_path).convert("RGBA")
        font = ImageFont.truetype("font.ttf", 40)
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        text_width, text_height = draw.textbbox((0, 0), watermark_text, font=font)[:2]
        position = (image.width - text_width - 10, image.height - text_height - 10) # Position the watermark in the lower right corner
        draw.text(position, watermark_text, font=font, fill=(128, 128, 128, 128))
        watermarked = Image.alpha_composite(image, overlay)
        watermarked.save(output_path)
        print(f"Watermarked image saved as {output_path}")
    except Exception as e:
        print(f"Error: {str(e)}")

def makeunique(new_unique_product_name):
    ai_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful budtender who knows all about the cannabis industry.",
            },
            {
                "role": "user",
                "content": f"Use this product name '{new_unique_product_name}'. Use this phrase to come up with a slightly different name that means the same thing."
            f"Come up with a new name that is max 70 chars long and will rank well with regard to SEO. If there is a mention of price. Change it to some other descriptive language instead."
            },
        ]
    )

def make_short_desc(current_short_description):
    # pdb.set_trace()
    ai_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful budtender who knows all about the cannabis industry.",
            },
            {
                "role": "user",
                "content": f"Use this product name '{source_product['short_description']}' to rewrite the product title."
            f"The new name is max 70 chars long and will rank well with regard to SEO."
            },
        ]
    )

def generate_new_product_name(sku):
    ai_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful budtender who knows all about the cannabis industry.",
            },
            {
                "role": "user",
                "content": f"Use this product name '{source_product['name']}' to rewrite the product title."
            f"The new name is max 70 chars long and will rank well with regard to SEO."
            },
        ]
    )
    new_product_name = ai_response['choices'][0]['message']['content'].strip()
    new_product_name = html.unescape(re.sub('<.*?>', '', new_product_name)).strip().replace('"','')
    return new_product_name

def generate_new_image_name(image_name):
    ai_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a creative AI assistant and California Budtender for a delivery service.",
            },
            {
                "role": "user",
                "content": f"I have an image with the name '{image_name}'. Need a new name for this image. Limit the name to 70 characters. Dont use any special punctuation."
            },
        ]
    )

    new_image_name = ai_response['choices'][0]['message']['content'].strip()
    new_image_name = html.unescape(re.sub('<.*?>', '', new_image_name))

    return new_image_name

def remove_keys(images_data):
    keys_to_remove = [
    'date_created',
    'date_created_gmt',
    'date_modified',
    'date_modified_gmt',
    'id',
    'alt'
    ]   
    new_images_data = []
    for index, image_data in enumerate(images_data):
        if index < 4:
            new_image_data = {key: value for key, value in image_data.items() if key not in keys_to_remove}
        else:
            new_image_data = {}
        new_images_data.append(new_image_data)
    return new_images_data







locations = []
def prompt_continue():
    print("   *Fix source product names at",location.capitalize(), "Doap before continuing...\n   (Press 'y' to continue or ctrl-c to cancel)")
    while True:
        key = input().strip()
        if key == " " or key.lower() == "y":
            return True
        elif key.lower() == "n":
            return False

with open(creds_file_path) as f:
    website = None
    user = None
    city = None
    phone = None
    consumer_key = None
    consumer_secret = None
    openai.api_key = None
    for line in f:
        line = line.strip()  # Remove trailing and leading whitespace
        if line.startswith("[") and line.endswith("]"):
            if website and user and city and phone and consumer_key and consumer_secret and openai.api_key:
                locations.append(Location(website, user, city, phone, consumer_key, consumer_secret, openai.api_key))
            website = line[1:-1].lstrip()  # Remove the brackets and any leading whitespace
            user = None
            city = None
            phone = None
            consumer_key = None
            consumer_secret = None
            openai.api_key = None
        elif website and " = " in line:
            key, value = line.split(" = ")
            if key == "user":
                user = value
            elif key == "city":
                city = value
            elif key == "phone":
                phone = value
            elif key.lower().endswith("_consumer_key"):
                consumer_key = value
            elif key.lower().endswith("_consumer_secret"):
                consumer_secret = value
            elif key == "openai.api_key":
                openai.api_key = value
                aikey = value
            elif key == "website":
                website = value
     
    locations.append(
        Location(website, user, city, phone, consumer_key, 
                 consumer_secret, openai.api_key)
    )
        
for locationa in locations[:1]:
    base_url = "https://" + locationa.website + "/wp-json/wc/v3/products"
    consumer_key = locationa.website + "_consumer_key:" + locationa.consumer_key
    consumer_secret = locationa.website + "_consumer_secret:" + locationa.consumer_secret
    aikey = openai.api_key
    auth = (
        locationa.consumer_key,
        locationa.consumer_secret,
    )
    #city = locationa.city
    #phone = locationa.phone
    #website = locationa.website
    response = requests.get(f'{base_url}', auth=auth, params={'sku': sku})
    response.raise_for_status()
    product = response.json()[0]
    source_product = product
    source_product['images'] = remove_keys(source_product['images'])
    source_images = source_product['images'][:4]  
    imagecounter = 0
    print("Source product title:\n",source_product['name'])
    print("\nSource images")
    for item in source_images:
        imagecounter = imagecounter + 1
        filename = os.path.basename(item['src'])
        print("#",imagecounter," Name: ", item['name'], " Filename: ",filename )
if not prompt_continue():
    sys.exit()

seq = 0

for locationb in locations[startfrom:]:
    seq = seq + 1
    base_url = "https://" + locationb.website + "/wp-json/wc/v3/products"
    consumer_key = locationb.website + "_consumer_key:" + locationb.consumer_key
    consumer_secret = locationb.website + "_consumer_secret:" + locationb.consumer_secret
    website = locationb.website
    subdomain = website.split('.')[0]
    #print("Get site # for ", subdomain, " ", get_site_id(subdomain))
    site_id = get_site_id(subdomain)
    print("Site ID:", site_id) 
    aikey = openai.api_key
    auth = (
        locationb.consumer_key,
        locationb.consumer_secret,
    )
    response = requests.get(f'{base_url}', auth=auth, params={'sku': sku})
    response.raise_for_status()
    product = response.json()[0]
    product['images'] = source_product['images'] 
    product['attributes'] = source_product['attributes'] 
    product['categories'] = source_product['categories'] 
    product['tags'] = source_product['tags'] 
    product['type'] = source_product['type'] 
    product['featured'] = source_product['featured'] 
    product['variations'] = source_product['variations'] 
    product['related_ids'] = source_product['related_ids'] 
    product['upsell_ids'] = source_product['upsell_ids'] 
    product['cross_sell_ids'] = source_product['cross_sell_ids'] 
    product['meta_data'] = source_product['meta_data'] 
    product['price'] = source_product['price'] 
    product['price_html'] = source_product['price_html'] 
    #product['brands'] = source_product['brands'] 
    pdb.set_trace()
    product['slug'] = source_product['slug'] 
    pdb.set_trace()
    if sku == '20665':
        product['short_description'] = new_short_description 
        # pdb.set_trace()
        product['description'] = new_description 
    else:
        print("No descriptions defined for that product")
        product['short_description'] = source_product['short_description'] 
        product['description'] = source_product['description'] 

    msgg = subdomain + " #" + str(seq) + " " + str(sku)
    print(msgg)
    # pdb.set_trace()
    while True:
        product_name = generate_new_product_name(sku).replace('"','').replace('"','').replace("'","").replace("_"," ").replace("(","").replace(")","").replace(",","").replace("$","")
        # print("Is this new product name okay?: ", product_name)
        # pdb.set_trace()
        print(product_name)
        choice = input("Do you want to use this new name [Y/N]: ")
        if choice.lower() == "y":
            product['name'] = product_name 
            break
    while True:
        choice = input("Do you want to update names of the existing images? [Y/N]: ")
        if choice.lower() == "y":
            #del product['images']
            print(subdomain + " existing images")
            imgcnt = 0
            for item in source_images:
                imgcnt = imgcnt + 1
                image_name = generate_new_image_name(item['name'])
                print("Image: ", imgcnt, " ", image_name, " ", item['src'])
                update_choice = input("Do you want to update this image? [Y/N]: ")
                if update_choice.lower() == "y":
                    # Code to update the image
                    item['name'] = image_name
                    print("New name assigned to image.", imgcnt)
                elif update_choice.lower() == "n":
                    print("Skipping image update.")
                else:
                    print("Invalid choice. Skipping image update.")
            break
        elif choice.lower() == "n":
            break

    print("Assigning new product name: ", product['name'])
    imagecounter = 0
    images = product['images']
    for item in images:
        imagecounter = imagecounter + 1
        filename = os.path.basename(item['src'])
        print("#",imagecounter," Name: ", item['name'], " Filename: ",filename )
        print("Assigning new image name: ", product['images'])
    #pprint.pprint(product['images'])
    print()
    update_url = f'{base_url}/{product["id"]}'
    update_response = requests.put(update_url, json=product, auth=auth)
    #pdb.set_trace()
    update_response.raise_for_status()
