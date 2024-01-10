import openai
import config
openai.organization = config.openai_organization
openai.api_key = config.openai_api_key

def translate(sentence, lang):
    text = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": f"Translate the given English sentence to {lang}. Respond by directly stating the translation in {lang}. Don't explain."
            },
            {
                "role": "user",
                "content": sentence
            },
        ],
        max_tokens=512,
        temperature=0, # the higher this value, the less deterministic
        top_p=1, # the higher this value, the wider range of vocab is used
    ).choices[0].message.content.strip()
    return text

if __name__ == "__main__":
    print(translate("She plays with fire by constantly challenging her boss 's authority", "Korean"))