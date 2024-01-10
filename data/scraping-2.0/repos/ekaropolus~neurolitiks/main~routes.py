from flask import Blueprint
from flask import render_template, request, url_for
# from mysite.models.models import Post
# from mysite import dash_app


main = Blueprint('main',__name__,template_folder='templates/', static_folder='static/')

import openai
# Set up OpenAI API credentials

@main.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index/index.html')



@main.route('/business/maker/', methods=['GET', 'POST'])
def index_a():
    if request.method == 'POST':
        roadmap_prompt = request.form['roadmap_prompt']
        roadmaps = generate_roadmap(roadmap_prompt)
    else:
        roadmaps = []
    return render_template('index/business_maker.html', roadmaps=roadmaps)

def generate_roadmap(roadmap_prompt):

    # Prompt for generating roadmap
    prompt = f"""Write in Spanish.
    Genera los pasos necesarios para que una PyME de {roadmap_prompt} genere sus estrategia para transformar su añadir a su empresa en secuencia sistemas informáticos, sistemas de analiticos y datos, y sistemas de inteligencia artifcial.
    Cada paso debe contener la estrategía y las tareas a realizar.
    Dame la respuesta en el siguiente formato, una linea por paso:
    paso x:"""
    # Generate text using OpenAI API
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=1000,
        n=1,
        stop=None,
        temperature=0.7
    )

    # Extract the generated roadmap from the API response
    roadmap_text = response.choices[0].text.strip()

    # Split the roadmap text into individual steps
    steps = roadmap_text.split('\n')

    roadmap = []
    for step in steps:
        step_lines = step.split('\n')
        step_info = {}

        for line in step_lines:
            if line.startswith("Paso") or line.startswith("paso"):
                step_info['description'] = step
                step_info['detailed_description'] = ""
                roadmap.append(step_info)
    return roadmap


# @main.route('/pp')
# def index_post():
#     page = request.args.get('page',1,type=int)
#     posts = Post.query.paginate(page = page, per_page = 1)
#     pages = [
#         {"name": "Details", "url": url_for("users.register", site_id="id")}
#     ]
#     return render_template("posts/posts_list.html", tittle = 'Home',posts=posts, pages=pages)




# # Define a Flask route for the Dash app
# @main.route('/dash')
# def dash_app_route():
#     return dash_app.index()




