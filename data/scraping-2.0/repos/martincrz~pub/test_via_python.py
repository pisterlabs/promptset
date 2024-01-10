import tkinter as tk
from PIL import Image, ImageTk
import requests
import io

import openai
openai.api_key = "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"

def generate_and_display(reseau, sujet, style, contenu, nb_mots):
    # Fermez la fenêtre modale
    modal.destroy()

    # Fonction qui génère le contenu et affiche la nouvelle fenêtre en plein écran
    new_window = tk.Tk()
    new_window.title("Résultat")

    screen_width = new_window.winfo_screenwidth()
    screen_height = new_window.winfo_screenheight()
    new_window.geometry(f"{screen_width}x{screen_height}")


    # On veut maintenant générer les posts publicitaires en fonction des paramètres
    # On va utiliser la librairie openai pour générer le texte et les images
    
    prompt = f"""Tu es un expert en communication sur {reseau}. 
Tu as été engagé par une entreprise pour créer une campagne publicitaire. Tu dois créer un post avec pour sujet {sujet} avec un style {style}.
    
La publication doit parler du produit ou service suivant : 
{sujet}
{contenu}
    
La publication doit bien respecter le formalisme du réseau et une taille de {nb_mots} mots. Ecris simplement la publication sur le service ou produit proposé, sans rien ajouter autour
    """

    print(f"Prompt publication : \n{prompt}\n\n")


    # response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",  # Sélectionnez le moteur OpenAI approprié
    #     messages=[{"role":"user", "content": prompt}],
    #     max_tokens= nb_mots,  # Définissez le nombre maximum de tokens pour la réponse
    #     temperature=1,
    #     n = 1 # spécifie le nombre de résultat
    # )
    # # Récupérez la réponse générée
    # generated_text = response.choices[0]["message"]["content"]
    generated_text= """🌱 Découvrez EcoFitGear, une gamme d'équipements de sport écologiques conçue pour les amateurs de sport soucieux de l'environnement. Nos produits sont fabriqués à partir de matériaux durables, recyclés ou écologiques, et sont conçus pour minimiser l'impact environnemental tout en offrant des performances de haute qualité. 🏃‍♂️

Dans notre gamme EcoFitGear, vous trouverez des chaussures de course durables, des vêtements de sport éco-responsables, de l'équipement de yoga en liège, des bouteilles d'eau"""
    print(f"Texte publication : \n{generated_text}\n\n")

    prompt_image =f"""Write a image prompt to generate an image image about the text below :
###
{generated_text}
###
    """
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",  # Sélectionnez le moteur OpenAI approprié
#         messages=[{"role":"system", "content": f"""You are a prompt engineer, spcialized in prompt image generation.
# here are some examples :
# 1- Woman, beautiful bedroom, glasses, showing skin, soft symmetric facial features, close up portrait, young, shot on sony a1, 85mm F/1. 4 ISO 100, medium format, 45 megapixel, flash lighting, natural sun lighting
# 2- photograph close up portrait old tough decorated general, serious, stoic cinematic 4k epic detailed 4k epic detailed photograph shot on kodak detailed bokeh cinematic hbo dark moody
# 3- photorealistic young mother walking at the beach, holding hands with her little son, mountains in the background, photography style, 85mm bookeh detailed, high resolution, high quality, natural lighting, ultra-detailed
                  
# The size and content of the image must respect the {reseau} formalisme. Write only the image prompt.
#                    """},
#                   {"role":"user", "content": prompt_image}],
#         max_tokens= 400,  # Définissez le nombre maximum de tokens pour la réponse
#         temperature=1,
#         n = 1 # spécifie le nombre de résultat
#     )
#     generated_prompt_image = response.choices[0]["message"]["content"]
    generated_prompt_image="A high-resolution image showcasing EcoFitGear products in a natural environment. The image depicts a scenic outdoor setting, with a lush green field in the foreground. In the center of the image, there is a display of various EcoFitGear items, including sustainable running shoes, eco-friendly sportswear, cork yoga equipment, and reusable water bottles. The products are arranged neatly on a wooden table, emphasizing their eco-conscious and durable nature. Sunlight softly illuminates the scene, highlighting the vibrant colors and sustainable materials used in EcoFitGear. The image captures the essence of environmentally friendly sports equipment while promoting an active and eco-conscious lifestyle."
    print(f"Prompt pour générer l'image : \n{generated_prompt_image}\n\n")

    # image= openai.Image.create(prompt=generated_prompt_image, n=1)
    # image_url = image["data"][0]['url']
    image_url = "https://oaidalleapiprodscus.blob.core.windows.net/private/org-bwG8HT8YW8RgPt1ndMNvjrzA/user-XyiYMI5r8OIZHqELN8TkyZjs/img-4mMFYu4zTptKchtX8bu98xaS.png?st=2023-10-23T08%3A44%3A21Z&se=2023-10-23T10%3A44%3A21Z&sp=r&sv=2021-08-06&sr=b&rscd=inline&rsct=image/png&skoid=6aaadede-4fb3-4698-a8f6-684d7786b067&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2023-10-22T17%3A51%3A05Z&ske=2023-10-23T17%3A51%3A05Z&sks=b&skv=2021-08-06&sig=quw88NyHUZBf0uFVj5lWPydRCGTNCA1/O1WeQzB0VMw%3D"

    
    HCTI_API_ENDPOINT = "https://hcti.io/v1/image"
    # Retrieve these from https://htmlcsstoimage.com/dashboard
    HCTI_API_USER_ID = '3b37f80b-765f-4b98-941b-58ccfcd21305'
    HCTI_API_KEY = 'd622874b-e12a-409e-bfb2-c27f279a9d48'

    if reseau=="LinkedIn":
        html = f"""
<div class="container">
  <div class="item infoRow">
    <div class='infoRow-Item infoRow-likedName font-bold'>Fujitsu</div>
    <div class='infoRow-Item, font-gray'>likes this</div>
    <div class="infoRow-Item infoRow-dotsMenu">...</div>
  </div>
  <div class="item">
    <div class='profileInfo'>
      <div class='profileInfo-image'></div>
      <div class='profileInfo-text'>
        <div>
          <span>WeeztR</span>
          <span class='profileInfo-nth font-gray'>2nd</span>
        </div>
        <div>
          <span class='font-gray'>Acteur de la transition écologique et défenseur du sport durable</span>
        </div>
      </div>
    </div>
    <div class='profileInfo-textSection'>
      <p>{generated_text}</p>

    </div>
    <div class='translationSection'>
      <span class='translationSection-translation font-bold'>See translation</span>
      <span>
        <span class='translationSection-likeIcon' />
        <span class='translationSection-heartIcon' />
        <span class='translationSection-clapIcon' />
        <span class='translationSection-likeCount'>115</span>
        <span>0 comments</span>
      </span>
    </div>
    <img src="{image_url}" alt="Image générée" style="width:100%; display:block;"/>
  </div>
  <div class="item">
    <div class='userActionSection'>
      <span class='userActionSection-icons userActionSection-like font-bold font-gray'>Like</span>
      <span class='userActionSection-icons userActionSection-comment font-bold font-gray'>Comment</span>
      <span class='userActionSection-icons userActionSection-share font-bold font-gray'>Share</span>
    </div>
  </div>
</div>"
"""
        css="""* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  background-color: #eee;
  font-family: -apple-system, system-ui, BlinkMacSystemFont, Segoe UI, Roboto,
    Helvetica Neue, Fira Sans, Ubuntu, Oxygen, Oxygen Sans, Cantarell,
    Droid Sans, Apple Color Emoji, Segoe UI Emoji, Segoe UI Symbol,
    Lucida Grande, Helvetica, Arial, sans-serif;
}

.container {
  background-color: white;
  width: 460px;
  margin: 10px auto;
  display: flex;
  flex-direction: column;
  border: 1px solid #d3d3d3;
}

.item {
  padding: 10px 20px;
  position: relative;
}

.item:first-of-type {
  border-bottom: 1px solid #d3d3d3;
}

.infoRow {
  display: flex;
  align-items: baseline;
}

.infoRow-likedName {
  margin: 0 4px;
}

.infoRow-dotsMenu {
  display: block;
  line-height: 0rem;
  font-size: 2rem;
  align-self: start;
  margin-left: auto;
}

.profileInfo {
  display: flex;
}

.profileInfo-image {
  flex: 0 1 60px;
  display: inline-block;
  width: 60px;
  height: 60px;
  border-radius: 60px;
  background-color: pink;
}

.profileInfo-text {
  flex: 1 1;
  display: flex;
  flex-direction: column;
  margin: 0 0 20px 10px;
}

.profileInfo-nth:before {
  content: " • ";
}

.profileInfo-textSection > p {
  margin-bottom: 1rem;
}

.profileInfo-more {
  display: inline;
}

.profileInfo-more:before {
  content: "\00a0";
  margin-left: 0.2rem;
}

.translationSection {
  display: flex;
  flex-direction: column;
}

.translationSection-translation {
  margin: 0.5rem 0;
  color: #006097;
}

.userActionSection:before {
  content: "";
  position: absolute;
  left: 20px;
  top: 2px;
  height: 1px;
  width: 410px;
  border-bottom: 1px solid #d3d3d3;
}

.translationSection-likeIcon:before {
  content: " 👍 ";
}

.translationSection-heartIcon:before {
  content: " ❤️ ";
}

.translationSection-clapIcon:before {
  content: " 👏 ";
}

.translationSection-likeCount:after {
  content: " • ";
}

.userActionSection-icons {
  margin-right: 0.5rem;
}

.userActionSection-like:before {
  content: " 👍 ";
}

.userActionSection-comment:before {
  content: " 💬 ";
}

.userActionSection-share:before {
  content: " ↗️ ";
}

.font-bold {
  font-weight: bold;
}

.font-gray {
  color: #777;
}"""
        data = { 'html': html,
         'css': css,
         'google_fonts': "Roboto" }
        

    # elif reseau == "Instagram":
    #     data = { 'html': "<div class='box'>Hello, world!</div>",
    #      'css': ".box { color: white; background-color: #0f79b9; padding: 10px; font-family: Roboto }",
    #      'google_fonts': "Roboto" }
    # elif reseau == "Facebook":
    #     data = { 'html': "<div class='box'>Hello, world!</div>",
    #      'css': ".box { color: white; background-color: #0f79b9; padding: 10px; font-family: Roboto }",
    #      'google_fonts': "Roboto" }
    # elif reseau == "Blog":
    #     data = { 'html': "<div class='box'>Hello, world!</div>",
    #      'css': ".box { color: white; background-color: #0f79b9; padding: 10px; font-family: Roboto }",
    #      'google_fonts': "Roboto" }

    print("\n\nHTML : \n")
    print(html)
    print("\n\n\n")
    image = requests.post(url = HCTI_API_ENDPOINT, data = data, auth=(HCTI_API_USER_ID, HCTI_API_KEY))
    print(image)
    print(image.json)
    
    print("Your image URL is : %s" %image.json()['url'])
    image_url = image.json()['url']

    # Utiliser Pillow (PIL) pour afficher l'image
    image_response = requests.get(image_url)
    image_data = image_response.content
    img = Image.open(io.BytesIO(image_data))
    img = img.resize((screen_width, screen_height), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)

    img_label = tk.Label(new_window, image=img)
    img_label.image = img
    img_label.pack()



def open_modal():
    global modal
    modal = tk.Toplevel(root)
    modal.title("NEW POST")

    # Ajoutez des éléments de formulaire, des boutons radio et d'autres éléments ici
    network_label = tk.Label(modal, text="Réseau social :")
    network_label.pack()

    networks = [("LinkedIn", "LinkedIn"), ("Facebook", "Facebook"), ("Instagram", "Instagram"), ("Blog", "Blog")]
    selected_network = tk.StringVar()
    selected_network.set("")
    for text, value in networks:
        tk.Radiobutton(modal, text=text, variable=selected_network, value=value).pack()

    sujet_label = tk.Label(modal, text="Sujet :")
    sujet_label.pack()
    sujet_entry = tk.Entry(modal)
    sujet_entry.pack()

    style_label = tk.Label(modal, text="Style :")
    style_label.pack()
    style_entry = tk.Entry(modal)
    style_entry.pack()

    contenu_label = tk.Label(modal, text="Contenu :")
    contenu_label.pack()
    contenu_text = tk.Text(modal, height=5, width=40)
    contenu_text.pack()

    caracteres_label = tk.Label(modal, text="Nombre de mots :")
    caracteres_label.pack()
    caracteres_slider = tk.Scale(modal, from_=0, to=500, orient="horizontal", length=200)
    caracteres_slider.pack()

    submit_button = tk.Button(modal, text="Generate", command=lambda: generate_and_display(selected_network.get(), sujet_entry.get(), style_entry.get(), contenu_text.get("1.0", tk.END), caracteres_slider.get()))
    submit_button.pack()



root = tk.Tk()
root.title("Jarvis - Génération de campagne")

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.geometry(f"{screen_width}x{screen_height}")

# Ruban vert avec le texte "Generative campaign" plus grand et aligné à gauche
ribbon = tk.Label(root, text="Generative campaign", bg="green", fg="white", font=("Helvetica", 24))
ribbon.pack(fill="x", anchor="w")

# Séparation entre le bouton et le ruban
separator = tk.Frame(root, height=150)
separator.pack(fill="x")

# Bouton "+" centré en vert, encore plus gros et parfaitement circulaire
center_button = tk.Button(root, text="+", bg="green", fg="white", font=("Helvetica", 48), width=3, height=1, relief="flat", command=open_modal)
center_button.pack()

# Marge entre le bouton et la barre de recherche
margin_between = tk.Frame(root, height=20)
margin_between.pack()

# Barre de recherche
search_bar = tk.Frame(root)
search_entry = tk.Entry(search_bar, width=30)
search_button = tk.Button(search_bar, text="Search", width=10)
search_entry.pack(side="left")
search_button.pack(side="right")
search_bar.pack()

root.mainloop()