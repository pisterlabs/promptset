import openai, json
class Extractor:
    def __init__(self, lang="Portugues", layout = None , numeric = False) -> None:
        self.lang = lang
        self.layout = layout
        self.numeric = numeric

    def extract(self, input, **kwargs)->dict:
        regras = [
            {"role": "system", "content": f"vou informar uma mensagem e você deve extrair as informaçoes e as mostrar em um json sem explicação nenhuma!, normalize as chaves"},
            {"role": "system", "content": f"As saidas devem estar na lingua {self.lang}"},
        ]
        if(self.numeric):
            regras.append({"role": "system", "content": f"se possivel trocar textos para valores numericos"})
        if not self.layout == None:
            regras.append({"role": "system", "content": f"use essa estrutura {json.dumps(self.layout)} json como modelo, valores faltantes prencha com \"\", corrija os campos cujo as informaçoes estão colocadas"})
            
        regras.append({"role": "system", "content": "agora vou enviar o texto"})
        regras.append({"role": "user", "content": input})

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages = regras
        )
        try:
            return json.loads(completion.choices[0].message.content)
        except:
            return {}

class IF():
    def __init__(self,) -> None:
        self.rules = []
        self.rules.append({"role": "system", "content": f"vamos fazer uma simulação de if vou informar uma mensagem e você deve dar o retorno correto sem explicação nenhuma!, caso a regra não exista mostre 'undefined', vou informar as regras"})
    
    def addRule(self ,condition="", response=""):
        self.rules.append({
            "role": "system",
            "content": f"se {condition} diga {response}"
        })
    
    def input(self, text)->str:
        rules = self.rules[:]
        rules.append({"role": "system", "content": "agora vou enviar o texto"})
        rules.append({"role": "user", "content": text})
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages = rules
        )
        return completion.choices[0].message.content
