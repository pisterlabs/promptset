import os

import openai
from rest_framework import serializers

from app.destinos.models import Destino
from app.destinos.validators import texto_descritivo_valido


class DestinoSerializer(serializers.ModelSerializer):
    class Meta:
        model = Destino
        fields = '__all__'

    def validate(self, data):
        if texto_descritivo_valido(data['texto_descritivo']):

            openai.api_key = os.getenv('OPENAI_API_KEY')

            prompt = f"Faça um resumo sobre {data['nome']} enfatizando o porque este lugar é incrível. Utilize uma linguagem informal e até 100 caracteres no máximo em cada parágrafo. Crie 2 parágrafos neste resumo"

            completion = openai.Completion.create(
                model='text-davinci-003',
                prompt=prompt,
                max_tokens=1024,
                temperature=0.5,
            )

            response = completion.choices[0].text

            data['texto_descritivo'] = response
        return data
