import openai



class gptController:
    def __init__(self,apiKey):
        openai.api_key  = apiKey

    def chatGPTrequest(self,text,slideCount,language):
        prompt = 'Make me a JSON file of a PowerPoint presentation of the theme "'+text+'" in '+language+'. This JSON file will contain multiple slides, each slide possibly containing the following: Title of the slide, search querys of images it may have, and possible text.Here is an example about deer:{"slides": [{"title": "Definición de venado","image": "deer","text": "Los venados son animales majestuosos que habitan en diversas regiones del mundo."},{"title": "Características del venado","image": "different species of deer","text": "Existen diferentes especies de venados, cada una con características únicas."},{"title": "Cuernos ramificados del venado","image": "deer horns","text": "Los venados poseen cuernos ramificados que son utilizados para la competencia y atracción de parejas."}, {"title": "Hábitats del venado","image": "A deer in a habitat","text": "Los venados se adaptan a diferentes hábitats, desde bosques hasta praderas y montañas."},{"title": "Importancia de la conservación del venado","image": "The importance of deer","text": "La conservación de los venados es crucial para preservar la biodiversidad y el equilibrio de los ecosistemas."}]}And so on, with multiple slides. Please note that:1. I want the image attribute to contain a simple search query in english of an image for pixabay not a URL.2.The image search query should not exceed 100 characters.3. The title attribute should contain the title of the slide, not the number of the slide.4. I want '+slideCount+' slides, no more, no less 5.I only want the JSON file in your response. Do not include any additional text, only JSON'
        response = openai.Completion.create(
            engine='text-davinci-003',
            prompt=prompt,
            max_tokens=3000,
            n=1,
            stop=None,
            temperature=0.7
        )
        if 'choices' in response and len(response['choices']) > 0:
            return response['choices'][0]['text']
        return {}
