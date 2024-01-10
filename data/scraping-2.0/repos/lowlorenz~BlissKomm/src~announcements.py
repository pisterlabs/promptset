import gradio as gr
import openai
import os
import prompt as p

openai.api_key_path = os.path.join(os.getcwd(), "openai_key.txt")

def get_openai_announcement(prompt):
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=prompt,
    temperature=0.9,
    max_tokens=300,
    top_p=1,
    frequency_penalty=0.0,
    presence_penalty=0.6,
    stop=["Task:"]
    )
    return response['choices'][0]['text']

def get_slack_announcement(date, paper, speaker, topic, institute):
    prompt = p.slack_speaker_prompt(date, paper, speaker, topic, institute)
    result = get_openai_announcement(prompt)
    return "Dear BLISSERS," + result

def get_twitter_announcement(date, paper, speaker, topic, institute):
    prompt = p.twitter_speaker_prompt(date, paper, speaker, topic, institute)
    result = get_openai_announcement(prompt)
    return result

def get_linkedin_announcement(date, paper, speaker, topic, institute):
    prompt = p.linkedin_speaker_prompt(date, paper, speaker, topic, institute)
    result = get_openai_announcement(prompt)
    return result

def build_announcement(date, paper, speaker, topic, institute):
    slack = get_slack_announcement(date, paper, speaker, topic, institute)
    twitter = get_twitter_announcement(date, paper, speaker, topic, institute)
    linkedin = get_linkedin_announcement(date, paper, speaker, topic, institute)
    return [slack, twitter, linkedin]

demo = gr.Interface(
    fn=build_announcement, 
    inputs=[gr.Text(label="Date"), gr.Text( label="Paper of the Week"),gr.Text(label="Speaker"), gr.Text(label="Topic"), gr.Text(label="Institute")], 
    outputs=[gr.Text(label="Slack"), gr.Text( label="Twitter"), gr.Text( label="LinkedIn")],
    title="Announcement Generator",
    examples=[
        ["May 30, 2023", "Drag Your GAN: Interactive Point-based Manipulation on the Generative Image Manifold", "Aray Karjauv", "Self-supervised representation learning using StyleGAN", "Dai Labor TU Berlin"],
        ["13.06", "Dopamine and temporal difference learning: A fruitful relationship between neuroscience and AI", "Nicolas Roth", "Computational Models of Visual Attention", "Neural Information Processing Group - TU Berlin"]
    ]
)

if __name__ == "__main__":
    demo.launch()