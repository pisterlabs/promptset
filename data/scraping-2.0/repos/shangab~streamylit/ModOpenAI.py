import streamlit as st
import openai
import ModUtils as ut

API_KEY = st.secrets["OPENAI_API_SECRET"]
openai.api_key = API_KEY


def completionsFunc(prompt, answer_area, error_area):
    with st.spinner("Please wait"):
        block = ''
        try:
            request = openai.Completion.create(
                engine='text-davinci-003', max_tokens=1000, prompt=prompt, stream=True)
            for resp in request:
                block += resp.choices[0].text
                answer_area.write(block)
        except:
            error_area.error('Error Happened')


def completions():
    st.title(":brain: GPT-3 Completions")
    st.subheader("GPT-3 Completions text-davinci-003 Engine")
    st.write(
        'GPT (Generative Pre-trained Transformer) is a type of language model developed by OpenAI that is trained on a large corpus of text using unsupervised learning techniques. One of the key features of GPT is its ability to generate coherent and meaningful text based on a given prompt or input. Here I am using streamlit library to call the [OpenAI API](https://openai.com/blog/openai-api)')
    st.write('OpenAI API is a language processing service that provides access to state-of-the-art machine learning models for natural language tasks.')
    st.write('Try the model below ')
    st.markdown("---")

    with st.expander("View Code", expanded=False):
        st.code(ut.getSource(completionsFunc))

    prompt = st.text_input('AI: What is up?')
    ans = st.empty()
    err = st.empty()

    if prompt:
        completionsFunc(prompt, ans, err)


def imageFunc(prompt, answer_area):
    with st.spinner("Please wait"):
        res = openai.Image.create(
            prompt=prompt,
            n=2,
            size="1024x1024")
    index = 1
    images_url = ''
    for one in res.data:
        images_url += f'[image {index}]({one.url}),  '
        index += 1
        st.image(one.url, width=400)
    answer_area.write(images_url)


def image():
    st.title(":brain: GPT-3 Image Genration")
    st.subheader(
        "GPT-3 Image Generation using GAN (Generative Adversarial Networks)")
    st.write("OpenAI uses a combination of deep learning algorithms, primarily generative adversarial networks (GANs), to generate images. GANs consist of two neural networks: a generator network and a discriminator network. The generator network takes random noise as input and generates images, while the discriminator network tries to distinguish between real images and generated images. The two networks are trained together in an adversarial process, where the generator tries to fool the discriminator, and the discriminator tries to correctly identify real images. As the training progresses, the generator learns to generate more realistic images that can fool the discriminator.")
    st.markdown("---")
    with st.expander("View Code", expanded=False):
        st.code(ut.getSource(imageFunc))
    prompt = st.text_input('Describe the image')
    ans = st.empty()

    if st.button("Generate Images!") and prompt:
        imageFunc(prompt, ans)


def classificationFunc(text, answer_area, error_area):
    with st.spinner("Please wait..."):
        prompt = f"""
            Text: The elections are coming\nClass: Politics\n\n
            Text: We will will the game\nClass: Sports\n\n
            Text: I will do my thesis\nClass: Academy\n\n
            Text: The universe is large\nClass: Science\n\n
            Text: chemistry is good\nClass: Science\n\n\n
            Text:{text}""",
        try:
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=prompt,
                temperature=0,
                max_tokens=60,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            st.write(response)
            answer_area.write(response.choices[0].text.strip())
        except:
            error_area.error('Error Happened')


def classification():
    st.title(":brain: GPT-3 Classification")
    st.subheader(
        "GPT-3  Classification")
    st.write("GPT-3's classification engine is a neural network that uses deep learning techniques to perform text classification. It is pre-trained on a large corpus of text and fine-tuned on specific classification tasks, such as sentiment analysis or topic classification. The model can predict the probability of a given text belonging to a particular category or class, and it can be further improved with additional training data.")
    st.write("Note the propmpt design to figure out how the text generation model works. This is what is called `Prompt Engineering` in GPT")
    st.markdown("---")
    with st.expander("View Code", expanded=False):
        st.code(ut.getSource(classificationFunc))
    text = st.text_area(
        'A text to classify from the belwo topics', value='We will win the second game for sure my friend.')
    ans = st.empty()
    err = st.empty()

    if st.button("Classify Article") and text:
        classificationFunc(text, ans, err)


def summarizationFunc(prompt, answer_area, error_area):
    with st.spinner("Please wait..."):
        try:
            request = openai.Completion.create(
                engine='davinci',
                max_tokens=60,
                prompt=prompt,
                n=1,
                stop=None,
                temperature=0.5,
                stream=False
            )
            if request.choices[0].text:
                answer_area.write(request.choices[0].text.strip())
            else:
                error_area.write("Error Happened")
        except:
            error_area.error('Error Happened')


def summarization():
    st.title(":brain: GPT-3 Summarization")
    st.subheader(
        "GPT-3 Summarization")
    st.write(
        'GPT (Generative Pre-trained Transformer) is a type of language model developed by OpenAI that is trained on a large corpus of text using unsupervised learning techniques. One of the key features of GPT is its ability to generate coherent and meaningful text based on a given prompt or input. Here I am using streamlit library to call the [OpenAI API](https://openai.com/blog/openai-api)')
    st.write('OpenAI API is a language processing service that provides access to state-of-the-art machine learning models for natural language tasks.')
    st.write('Try the model below ')
    st.markdown("---")
    with st.expander("View Code", expanded=False):
        st.code(ut.getSource(summarizationFunc))
    prompt = st.text_area('A text to summarize please:')
    ans = st.empty()
    err = st.empty()

    if st.button("Summarize Article") and prompt:
        summarizationFunc(prompt, ans, err)
