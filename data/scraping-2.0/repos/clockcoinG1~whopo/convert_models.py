import openai
models = openai.Model.list()
models_str = '\n'.join(models)
with open('models.txt', 'w') as f:
    f.write(models_str)