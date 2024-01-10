import requests
import json
import openai
from get_triplets import get_relevant_pairs
class utilities:
    @staticmethod
    def data_to_list_ploty(data : dict):
        labels = []
        values = []
        parents = []
        for aspect, opinions_sent in data.items():
            labels.append(aspect)
            parents.append('')
            values.append(sum(opinions_sent.values()))
            for opinion, count in opinions_sent.items():
                labels.append(opinion)
                parents.append(aspect)
                values.append(count)
        return labels, parents, values
    

    @staticmethod
    def get_triplets(review : str):
        url_ngrok = "http://3904-34-172-214-75.ngrok-free.app"
        url_colab = url_ngrok+f"/api/predict?review='{review}'"
        response = requests.get(url_colab)
        return response.json()


    @staticmethod
    def get_relevant_pairs(triplets : list):
        phrase_pairs = [(triplet['aspect'], triplet['opinion']) for triplet in triplets]
        relevant_pairs = get_relevant_pairs(phrase_pairs)
        return relevant_pairs
    
    @staticmethod
    def generate_review(relevant_pairs : list):
        openai.api_key = "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"
        promt = f"""
        Escribe una reseña para tu página que incluya la siguiente lista de pares de aspectos y opiniones, 
        donde la opinion es un adjetivo que califica al aspecto, utiliza lenguaje claro para transmitir tu experiencia de forma casual.
        A continuación los pares:
        {str(relevant_pairs)}
        """
        print(promt)
        response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
                    {"role": "system", "content": """
                        Eres Monchi un critico de restaurantes, 
                        y que sus reseñas son de un maximo de 1000 caracteres, cada reseña comienza con un titulo corto,
                        que resume la experiencia general.
                    """.strip().replace("\n", " ").replace("\t", " ")},
                    {"role": "user", "content": """
                        Escribe una reseña para tu página que incluya la siguiente lista de pares de aspectos y opiniones, es decir [aspecto, opinion], donde la opinion es un adjetivo que califica al aspecto, utiliza lenguaje claro para transmitir tu experiencia de forma casual.
                        A continuación los pares:
                        [['restaunte', 'bueno'], ['servicio', 'atento'], ['ambiente', 'agradable'], ['chilaquiles', 'deliciosos'], ['huevos rancheros', 'pesimos'], ['tacos dorados', 'buenos'], ['platillos', 'mal olor'], ['bebidas', 'deliciosas'], ['bebidas', 'caras']]
                    """.strip().replace("\n", " ").replace("\t", " ")},
                    {"role": "assistant", "content": """
                        Una joya escondida con algunos contratiempos

                        Encontré un restaurante bueno al que tenía ganas de probar. Puedo confirmar que el servicio fue atento durante toda nuestra visita, y el ambiente en general era agradable y cómodo. En cuanto a la comida, los chilaquiles se destacaron como una opción deliciosa y te los recomendaría a cualquiera. Sin embargo, los huevos rancheros dejaron mucho que desear y resultaron ser pésimos.

                        Afortunadamente, los tacos dorados lograron compensar esa mala experiencia, y calificaría estos como buenos. A pesar de eso, hubo un detalle que no pude ignorar: algunos platillos tenían un mal olor que hacía cuestionar su calidad. Por otro lado, las bebidas en este lugar eran deliciosas, aunque también me parecieron algo caras.

                        En resumen, este restaurante tiene un potencial increíble, pero es necesario prestar atención a ciertos detalles antes de que pueda ser considerado uno de los mejores de la ciudad. De momento, lo recomendaría con precaución y seleccionando muy bien lo que se decide pedir en el menú.
                    """.strip().replace("\t", " ")}, 
                    {"role": "user", "content": promt}
                ]
        )
        print(response['choices'][0]['message']['content'])
        return response['choices'][0]['message']['content']
    
    @staticmethod
    def from_triplets_to_db(triplets : list) -> dict:
        data = {}
        for triplet in triplets:
            aspect = triplet['aspect']
            opinion = triplet['opinion']
            if aspect not in data:
                data[aspect] = {}
            if opinion not in data[aspect]:
                data[aspect][opinion] = 0
            data[aspect][opinion] += 1
        return data
    
    """
    let triplets = [
        [[4], [1, 2], 'POS'],
        [[35, 36, 37], [38], 'POS'],
        [[41, 42, 43, 44, 45], [48], 'POS'],
        [[52], [51], 'POS'],
        [[54], [58], 'POS'],
        [[54], [61], 'POS']
    ];
    """
    


    @staticmethod
    def from_db_to_triplets_js(reviews_db : dict) -> list:
        reviews_triplets = []
        #make fill_positions a lambda function with argument start_end_positions
        fill_positions = lambda start_end_positions : [i for i in range(start_end_positions[0], start_end_positions[1]+1)]
        for review_db in reviews_db:
            triplets = []
            for triplet_db in review_db['triplets']:
                triplets.append([
                    fill_positions(triplet_db['positions']['aspect']),
                    fill_positions(triplet_db['positions']['opinion']),
                    triplet_db['sentiment']])
            reviews_triplets.append(dict(triplets=triplets, review=review_db['review']))
        return reviews_triplets
    
    @staticmethod
    def get_aspects(reviews_triplets : list) -> list:
        aspects = []
        for review_triplets in reviews_triplets:
            for triplet in review_triplets['triplets']:
                aspects.append(triplet['aspect'])
        return aspects
    
    @staticmethod
    def get_opinions(reviews_triplets : list) -> list:
        opinions = []
        for review_triplets in reviews_triplets:
            for triplet in review_triplets['triplets']:
                opinions.append(triplet['opinion'])
        return opinions