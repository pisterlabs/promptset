# 📚 Basic libraries
import os # file managment
import numpy as np # image array manipulation
from PIL import Image # image data visualization module from Python Imaging Library

# 🌐 Computer Vision
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# 🛠️ Tools
import openai # chat-gpt3 API
import streamlit as st # THE library to make the app demo

# API key for GPT-3
openai.api_key = 'YOUR-API-KEY'

# page configuration
st.set_page_config(page_title='Cytology Codex', layout='wide')
top_sidebar_placeholder = st.sidebar.empty()

# image for Home page
top_sidebar_placeholder.markdown('''
<p align="center">
  <img src="https://i.postimg.cc/0NJXSKtL/cytology-codex-final.png" width="100%" alt="Cytology Image">
  <br>
</p>
''', unsafe_allow_html=True)

# sidebar
st.sidebar.title('Navigation')
page = st.sidebar.radio('Select a Page', ['Home', 'Diagnosis', 'About'])
about_selection = ''

with st.spinner('Loading page...'):
    if page == 'Home':
        st.title('Cytology Codex')
        st.subheader('The Digital Tome of Cytology: Deep Learning & Neural Networks Transmutation')
        st.markdown('''
            <p align="center">
              <img src="https://imgtr.ee/images/2023/07/14/b1deb27bdd471bb2b8fc5374d9d9cca5.png" width="36%" alt="Cytology Image">
              <br>
              <small style="font-size:18px;"><em>Behold, the Digital Tome of Cytology, revealing microscopic marvels and cellular untold tales.</em></small>
            </p>
            ''', unsafe_allow_html=True)

                
    elif page == 'Diagnosis':
        # Title and logo
        st.markdown("# Diagnose and Assistance")

        # data path
        data_path = os.path.join('C:\\Users\\apisi\\01. IronData\\01. GitHub\\03. Projects\\08_cells_at_work', '01_data')

        # salivary gland model
        salivary_gland = os.path.join(data_path, '01_salivary_gland')
        checkpoint_path = os.path.join(salivary_gland, "00_epochs", "save_at_30.keras")
        salivary_model = tf.keras.models.load_model(checkpoint_path)

        # gynecological gland model
        gynecological = os.path.join(data_path, '02_gynecological')
        checkpoint_path2 = os.path.join(gynecological, "00_epochs", "save_at_27.keras")
        gynecological_model = tf.keras.models.load_model(checkpoint_path2)

        # thyroid gland model
        thyroid = os.path.join(data_path, '03_thyroid')
        checkpoint_path3 = os.path.join(thyroid, "00_epochs", "save_at_28.keras")
        thyroid_model = tf.keras.models.load_model(checkpoint_path3)

        # effusions model
        effusions = os.path.join(data_path, '04_efussions_wellgen')
        checkpoint_path4= os.path.join(effusions, "00_epochs", "save_at_25.keras")
        effusions_model = tf.keras.models.load_model(checkpoint_path4)

        # mapping between the select box options and the models
        models = {
            "Salivary Gland": salivary_model,
            "Gynecological": gynecological_model,
            "Thyroid": thyroid_model,
            "Effusions": effusions_model,
        }

        # classes dictionary per model
        classes = {
            "Salivary Gland": {i: name.replace('_', ' ').title() for i, name in enumerate(['acinar_carcinoma', 'adenoid_cystic_carcinoma', 'pleomorphic_adenoma', 'warthin_tumor'])},
            "Gynecological": {i: name.upper().replace('_', '-') for i, name in enumerate(['h-sil', 'l-sil'])},
            "Thyroid": {i: name.replace('_', ' ').title() for i, name in enumerate(['papillary_carcinoma', 'chronic_thyroidits'])},
            "Effusions": {i: name.replace('_', ' ').title() for i, name in enumerate(['Positive', 'Negative'])}, 
        }

        # Select organ sample using selectbox
        organ_sample = st.selectbox('', ['Salivary Gland', 'Gynecological', 'Thyroid', 'Effusions'])

        # Image upload
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])


        if uploaded_file is not None:
            img = Image.open(uploaded_file).convert("RGB")
            img_resized = img.resize((256, 256))
            img_array = np.expand_dims(image.img_to_array(img_resized), axis=0)

            # Load the model and classes dictionary for the chosen organ sample
            model = models[organ_sample]
            class_dict = classes[organ_sample]

            # Progress bar
            progress_bar = st.progress(0)
            for i in range(100):
                progress_bar.progress(i + 1)
            st.text("Complete!")

            with st.spinner('Predicting...'):
                predictions = model.predict(img_array)
                class_idx = np.argmax(predictions[0])
                class_name = class_dict[class_idx]

            # Result presentation
            col1, col2 = st.columns(2)
            with col1:
                col1.markdown(f"<h2 style='text-align: center; color: aqua;'>Image suggestive of: {class_name}</h2>", unsafe_allow_html=True)
                col1.image(img, use_column_width=True)

            with col2:
                # Button queries
                col2.markdown('### Cytology Consultant')
                button_queries = {
                    "Useful IHQ": f"What are the most frequently used immunohistochemical markers for {class_name} in Cytology?",
                    "Differential Diagnosis": f"What are the differential diagnoses for {class_name} in Cytology?",
                    "Useful books": f"What are the best books for {class_name} in Cytology?"
                }

                # Button text
                cols = col2.columns(len(button_queries))  # create columns for each button
                for i, (button_text, query) in enumerate(button_queries.items()):
                    if cols[i].button(button_text):
                        with st.spinner('Consulting with AI...'):
                            response = openai.ChatCompletion.create(
                                model="gpt-3.5-turbo",
                                messages=[
                                    {"role": "system", "content": "You are a helpful assistant."},
                                    {"role": "user", "content": query}
                                ],
                            )
                        col2.markdown(f"{response['choices'][0]['message']['content']}")

                # AI response for cytological features and malignancy risk
                with st.spinner('Consulting with AI...'):
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": f"In 200 words length, which are the key cytological features of {class_name} in the {organ_sample}, and what is its risk of malignancy?"}
                        ],
                    )
                col2.markdown(f"{response['choices'][0]['message']['content']}")

                st.markdown("###### AI powered by Chat GPT-3", unsafe_allow_html=True)
        
    elif page == 'About':
        about_selection = st.sidebar.radio('Select a Topic', ['Cytology', 'Data', 'Neural Networks'])
        if about_selection == 'Cytology':
            st.markdown('''
            <h1 id="about-cytology">About Cytology</h1>
            <p align="center">
              <img src="https://cdn.discordapp.com/attachments/1129416761922568212/1130370346319609926/isi.mube_an_open_spellbook_featuring_in_their_pages_cytology_im_e604e964-a907-453f-b855-4d34c82382be.png" width="20%">
            </p>
            <h2>Glossary</h2>
            <p align="justify">Let´s define first a few key terms:</p>
            <ul>
                <li><strong>Cytology:</strong> This is the study of individual cells to detect abnormalities, including cancer. It's a type of sample method that provides a less invasive alternative to biopsies, enabling early diagnosis and treatment initiation, and improving health outcomes.</li>
                <li><strong>Cytopathology:</strong> A specialized field, nested in pathology, that looks at diseases on the cellular level. Professionals of cytopathology include Cytotechnologists & Cytopathologists, focusing on screening, interpretation, and diagnosis of diverse cell samples.</li>
                <li><strong>Digital Pathology:</strong> This involves digitizing pathology slides, allowing the use of image-based information for diagnosis, research, and teaching. Digital Pathology includes not only the digitalization of histology and cytology slides but also the automatization, technology, and tools of all preanalytical, analytical and post-analytical processes in a pathology department</li>
            </ul>

            <h2>Challenges in Digital Cytology</h2>
            <p align="justify">In digital cytology, we face a unique challenge. Unlike in histology, where cells maintain their flat structure (like a single layer of bricks on a wall), cytology samples can be more like a pile of bricks dumped out of a bucket. These cells in suspension no longer hold their original formation, making diagnosis more complex and time-consuming because it requires mastery of pattern recognition. Furthermore, due to these additional dimensions, digitizing these cell images requires even more storage space.</p>
            <br>
            <p align="center">
            <img src="https://i.postimg.cc/YCxYsrvh/cytology-codex-final-1.png" width="26%">
            <br>
            <small><em>Thyroid, papillar carcinoma. Same tumor, different methods and different features. On the left, histology (1-dimensional thin layer), and on the right, cytology (three-dimensional in suspension cells).</em></small>
            </p>

            <h2>Personal Journey and Perspectives on Cytology</h2>
            <p align="justify">My past 5 years of work have been all around Cytology; it involved screening and diagnosis of numerous cytology specimens, quality control, and engaging in both teaching and research, including Digital Pathology publications.</p>

            <p align="justify">One significant barrier to the digitalization of Cytological samples is the final size. As previously explained, the cells in Cytology are not flat, unlike in Histology, but three-dimensional. This complexity typically necessitates a Z-stack scanning of the slides to capture all focal points, resulting in large digital files.</p>

            <p align="justify">Despite this challenge, I firmly believe that Machine Learning and Deep Learning models can be implemented in Cytology images, bypassing the need for complete scan, hence one of the most challenging aspects of the digitalization process.</p>
            ''', unsafe_allow_html=True)
                    
        if about_selection == 'Data':
            st.markdown('''
            <div>
            <h1 id="about-the-data">About the Data</h1>
            <p align="center">
              <img src="https://cdn.discordapp.com/attachments/1129416761922568212/1130370155399094293/isi.mube_an_open_spellbook_featuring_in_their_pages_data_charts_d4f6523e-ff38-4e94-a11c-cd151d7087e2.png" width="20%">
            </p>
            <p style="text-align:justify">Gathering and extracting sufficient cytology image data for the project was a challenging step due to the lack of publicly available images, slides, and online microscopes.</p>

            <h2 id="first-idea">First Idea</h2>

            <p style="text-align:justify">The original idea was to use my own data, given my direct access to it as a professional in the field. However, the process of manually selecting and capturing the images using my phone camera and microscope proved to be time-consuming and would have necessitated extensive preprocessing. This was not the best approach for an ambitious three-week project.</p>

            <div style="display: flex; justify-content: center;">
            <div style="width: 30%; padding: 1%;">
            <p align="center">
                <img src="https://i.postimg.cc/Xq9wg1Q9/data-extraction.png" style="width: 100%;">
            </p>
            </div>
            <div style="width: 30%; padding: 1%;">
            <p align="center">
                <img src="https://i.postimg.cc/sXhSV0Fb/data-extraction2.png" style="width: 100%;">
            </p>
            </div>
            </div>

            <h2 id="second-idea">Second Idea</h2>

            <p style="text-align:justify">Another option I considered was using images from book atlases and other resources I had accumulated over my years of practice. While these images provided a potential solution, their limited variety and quantity posed challenges for training a robust model.</p>

            <p align="center">
            <img src="https://i.postimg.cc/W4WGXkjb/books.png" width="23%">
            </p>

            <h2 id="final-idea">Final Idea</h2>

            <p style="text-align:justify">Ultimately, a breakthrough idea came from one of my best friends and colleagues, <a href="https://www.linkedin.com/in/xose-fern%C3%A1ndez-5a8064a3?originalSubdomain=es">Xose Fernánndez</a>, a fellow Cytotechnologist and Cytology Professor. He granted me access to an online microscope they use for teaching purposes, with a diverse range of scanned slides featuring different organs and diagnoses. After conducting research on their platform, I identified the organs and diagnoses that could provide the most data.</p>

            <p style="text-align:justify">My initial data collection focused primarily on salivary gland samples, given the rich archive of slides and diagnostic diversity. Using my domain knowledge, I manually selected the fields of interest, employing the criteria used in Cytopathology for diagnosis. For instance, in the case of a Warthin Tumor, it was important to identify features such as a "dirty background" with lymphocytes and oncocytic cells.</p>

            <p style="text-align:justify">Despite these efforts, the available data was still limited. To address this challenge, I used data augmentation techniques in Python to generate synthetic images, thereby augmenting the training and validation datasets.</p>

            <p align="center">
            <img src="https://i.postimg.cc/PJDBQHgK/warthin.png" width="36%">
            <br>
            <small><em>Synthetic Data of a Warthin Tumor, generated by Keras using ImageDataGenerator.</em></small>
            </p>

            <h2 id="final-notes">Final Notes</h2>

            <p style="text-align:justify">Due to privacy concerns, I did not upload the final image folders to GitHub. Although ethical guidelines within academic cytology societies (of which I am a member) permit the use of individual images for research, education, and promoting cytology, strict protocols are followed in the medical community to ensure confidentiality and anonymity.</p>
            </div>
            ''', unsafe_allow_html=True)

        if about_selection == 'Neural Networks':
            st.markdown('''
            <h1 id="about-neural-networks">About Neural Networks, Deep Learning and Image Multiclassification</h1>
            <p align="center">
              <img src="https://cdn.discordapp.com/attachments/1129416761922568212/1130370214043861022/isi.mube_an_open_spellbook_featuring_in_their_pages_neural_netw_397a026c-43bb-4da0-b8c1-ffe76f4b21eb.png" width="20%">
            </p>
            <p>To keep it nice, clean and easy to read, my code is structured in two major scripts which are designed to be reusable for different organ samples in an iterative process.</p>
            ''', unsafe_allow_html=True)

            st.header('Python Script I: Data Extracting, Cleaning and Pre-Processing')
            st.markdown('''
            The first script is responsible for data extraction, cleaning and preprocessing, which involves creating folder names with OS for different diagnoses per organ, training and validation subsets, copying images, and applying data augmentation techniques.
            ''')
            st.code('''
            # Data Augmentation Parameters
            datagen = ImageDataGenerator(
                rescale=1.0/255,  # normalizes pixel values to [0, 1]
                rotation_range=30, # cells can appear at any orientation
                width_shift_range=.15, # cells can be located anywhere
                height_shift_range=.15, # same as before, but vertically
                horizontal_flip=True, # cell orientation doesn't matter
                vertical_flip=True, # cell orientation doesn't matter
                brightness_range=[0.5, 1.5],  # simulates variable lighting/staining
                zoom_range=0.2  # simulates variable cell sizes/distances
            )
            ''', language='python')

            st.markdown('I also saved 15% of the original data as unique copies for testing purposes.')

            st.header('Python Script II: Deep Learning Modeling, Validation and Testing')
            st.markdown('''
            The second script is dedicated to building and validating the Deep Learning model, as well as testing its performance. I'll briefly explain the structure of the Convolutional Neural Network (CNN) used in this project:
            ''')

            st.code('''
            x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
            x = layers.MaxPooling2D((2, 2))(x)

            x = layers.Conv2D(64, (3, 3), activation='relu')(x)
            x = layers.MaxPooling2D((2, 2))(x)

            x = layers.Conv2D(128, (1, 1), activation='relu')(x)
            x = layers.MaxPooling2D((2, 2))(x)

            x = layers.Conv2D(256, (2, 2), activation='relu')(x)
            x = layers.MaxPooling2D((2, 2))(x)

            x = layers.Flatten()(x)
            x = layers.Dense(512, activation='relu')(x)
            x = layers.Dense(256, activation='relu')(x)
            x = layers.Dense(5, activation='softmax')(x)

            return keras.Model(inputs, x)
            ''', language='python')
            st.subheader('Conv2 Layer')

            st.markdown('''
            Convolutional layer of the network, where the image pre-processing happens and the kernel filter the image on the whole image, detecting lines and edges. '32', '64', '128' refers to the number of filters used, starting from a small number. 'Relu' activation adds non-linearity to the model, to learn more complex patterns.
            ''')

            st.markdown('<p align="center"><img src="https://i.postimg.cc/KjMcCdMs/cytology-codex-final-6.png" width="36%"></p>', unsafe_allow_html=True)

            st.subheader('MaxPooling2D Layer')

            st.markdown('''
            It performs downsampling operations, and also spatial dimensions (width, height), reducing overfitting and the computational cost by decreasing the spatial dimensionality. This process goes along with the convolutional layer, with different kernel sizes, to learn more complex patterns from Data.
            ''')

            st.markdown('<p align="center"><img src="https://i.postimg.cc/J4jj9hjm/cytology-codex-final-7.png" width="36%"></p>', unsafe_allow_html=True)

            st.subheader('Flatten Layer')

            st.markdown('''
            It prepares the multi-dimensional input for the last fully and dense layer, converting the 2D matrix to a 1D vector.
            ''')

            st.markdown('<p align="center"><img src="https://i.postimg.cc/Wb7hGqMD/cytology-codex-final-8.png" width="36%"></p>', unsafe_allow_html=True)

            st.subheader('Dense Layer')

            st.markdown('''
            It takes the Flatten inputs through a series of neurons. First 512, then 256. It's where all previous neurons are fully connected, the 'softmax' activation it's used in multi-classification models, in this case, '5' corresponds to the number of target classes.
            ''')

            st.markdown('<p align="center"><img src="https://i.postimg.cc/jS5dygx9/cytology-codex-final-9.png" width="36%"></p>', unsafe_allow_html=True)

            st.markdown('''
            In short, this CNN model learns hierarchical representations of the images, from low-level structures (lines, edges, textures) in the initial layers to more complex high-level features such as patterns and objects.
            ''')