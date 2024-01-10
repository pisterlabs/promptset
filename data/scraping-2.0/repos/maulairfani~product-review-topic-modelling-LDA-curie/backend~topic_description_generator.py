import openai

def generate_prompt(nama_produk, jumlah_rating, topic):
    prompt = {
    'nama_produk': nama_produk,
    'jumlah_rating': jumlah_rating,
    'topic' : topic
    }
    prompt = str(prompt)[17:-3] + ' ->'

    return prompt

def generate_text(nama_produk, jumlah_rating, topic, api_key = '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'):
    openai.api_key = api_key

    prompt = generate_prompt(nama_produk, jumlah_rating, topic)
    completion = openai.Completion.create(
        engine='curie:ft-personal:review-product-fine-tuned-model-2023-01-11-15-03-00',
        prompt=prompt,
        max_tokens=700,
        n=1,
        temperature=0,
        stop=[". END"]
    )
    return completion