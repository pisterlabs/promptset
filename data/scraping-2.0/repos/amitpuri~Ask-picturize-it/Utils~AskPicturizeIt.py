from langchain.prompts import PromptTemplate

class AskPicturizeIt:
    TITLE = '# [Ask-picturize-it](https://github.com/amitpuri/Ask-picturize-it)'
    DESCRIPTION = """<strong>This space uses following:</strong>
       <p>
       <ul>    
       <li>OpenAI API</li>    
           <ul>    
               <li>Whisper(whisper-1) <a href='https://openai.com/research/whisper'>https://openai.com/research/whisper</a></li>    
               <li>DALL-E <a href='https://openai.com/product/dall-e-2'>https://openai.com/product/dall-e-2</a></li>
               <li>GPT <a href='https://openai.com/product/gpt-4'>https://openai.com/product/gpt-4</a></li>  
           </ul>
       <li>Azure OpenAI <a href='https://azure.microsoft.com/products/cognitive-services/openai-service'>https://azure.microsoft.com/products/cognitive-services/openai-service</a></li>  
       <li>Google Generative AI (PaLM API)<a href='https://developers.generativeai.google'>https://developers.generativeai.google</a></li>         
       <li>Cloudinary <a href='https://cloudinary.com/documentation/python_quickstart'>https://cloudinary.com/documentation/python_quickstart</a></li>
       <li>Gradio App <a href='https://gradio.app/docs'>https://gradio.app/docs</a> in Python and MongoDB</li>
       <li>Prompt optimizer <a href='https://huggingface.co/microsoft/Promptist'>https://huggingface.co/microsoft/Promptist</a></li>
       <li>stabilityai/stable-diffusion-2-1 <a href='https://huggingface.co/stabilityai/stable-diffusion-2-1'>https://huggingface.co/stabilityai/stable-diffusion-2-1</a></li>
       <li>Stability AI <a href='https://stability.ai'>https://stability.ai</a></li>
       <li>LangChain OpenAI <a href='https://python.langchain.com/en/latest/modules/models/llms.html'>https://python.langchain.com/en/latest/modules/models/llms.html</a></li>
       <li>Article Extractor and Summarizer on Rapid API <a href='https://rapidapi.com'>https://rapidapi.com</a></li> 
       <li>A Python package to assess and improve fairness of machine learning models.<a href='https://fairlearn.org'>https://fairlearn.org</a></li>  
       </ul>
       </p>
     """
    RESEARCH_SECTION = """


       <p><strong>Check it out</strong></p>
       <p>
       <ul>

       <li><p>Attention Is All You Need <a href='https://arxiv.org/abs/1706.03762'>https://arxiv.org/abs/1706.03762</a></p></li>
       <li><p>NLP's ImageNet moment has arrived <a href='https://thegradient.pub/nlp-imagenet'>https://thegradient.pub/nlp-imagenet</a></p></li>   
       <li><p>Zero-Shot Text-to-Image Generation <a href='https://arxiv.org/abs/2102.12092'>https://arxiv.org/abs/2102.12092</a></p></li>   
       <li><p>Transformer: A Novel Neural Network Architecture for Language Understanding <a href='https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html'>https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html</a></p></li>
       <li><p>CS25: Transformers United V2 <a href='https://web.stanford.edu/class/cs25'>https://web.stanford.edu/class/cs25</a></p></li>
       <li><p>CS25: Stanford Seminar - Transformers United 2023: Introduction to Transformer <a href='https://youtu.be/XfpMkf4rD6E'>https://youtu.be/XfpMkf4rD6E</a></p></li>
       <li><p>Temperature in NLP <a href='https://lukesalamone.github.io/posts/what-is-temperature'>https://lukesalamone.github.io/posts/what-is-temperature</a></p></li>
       <li><p>openai-cookbook <a href='https://github.com/openai/openai-cookbook'>https://github.com/openai/openai-cookbook</a></p></li>                     
       <li><p>High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs <a href='https://tcwang0509.github.io/pix2pixHD'>https://tcwang0509.github.io/pix2pixHD</a></p></li>
       <li><p>Denoising Diffusion Implicit Models <a href='https://keras.io/examples/generative/ddim/'>https://keras.io/examples/generative/ddim/</a></p></li>
       <li><p>A walk through latent space with Stable Diffusion <a href='https://keras.io/examples/generative/random_walks_with_stable_diffusion'>https://keras.io/examples/generative/random_walks_with_stable_diffusion</a></p></li>
       <li><p>DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation <a href='https://dreambooth.github.io'>https://dreambooth.github.io</a></p></li>              
       <li><p>Alpaca Eval Leaderboard <a href='https://tatsu-lab.github.io/alpaca_eval'>https://tatsu-lab.github.io/alpaca_eval</a></p></li>
       <li><p>Open LLM Leaderboard <a href='https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard'>https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard</a></p></li>
       <li><p>LLM Sys projects <a href='https://lmsys.org/projects'>https://lmsys.org/projects</a></p></li>
       <li><p>A list of open LLMs available for commercial use<a href='https://github.com/eugeneyan/open-llms'>https://github.com/eugeneyan/open-llms</a></p></li>
       <li><p>Aviary Explorer  <a href='https://aviary.anyscale.com'>https://aviary.anyscale.com</a></p></li>       
       <li><p>LangChain Python <a href='https://python.langchain.com'>https://python.langchain.com</a></p></li>
       <li><p>LangChain for Gen AI and LLMs <a href='https://www.youtube.com/playlist?list=PLIUOU7oqGTLieV9uTIFMm6_4PXg-hlN6F'>https://www.youtube.com/playlist?list=PLIUOU7oqGTLieV9uTIFMm6_4PXg-hlN6F</a></p></li>
       <li><p>LangChain's integration with Chroma <a href='https://blog.langchain.dev/langchain-chroma'>https://blog.langchain.dev/langchain-chroma</a></p></li>
       <li><p>Vector Similarity Explained <a href='https://www.pinecone.io/learn/vector-similarity'>https://www.pinecone.io/learn/vector-similarity</a></p></li>
       <li><p>Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks <a href='https://arxiv.org/abs/2005.11401'>https://arxiv.org/abs/2005.11401</a></p></li>
       <li><p>An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <a href='https://arxiv.org/abs/2010.11929'>https://arxiv.org/abs/2010.11929</a></p></li>
       <li>stable-diffusion-image-variations <a href='https://huggingface.co/spaces/lambdalabs/stable-diffusion-image-variations'>https://huggingface.co/spaces/lambdalabs/stable-diffusion-image-variations</a></li> 
       <li>text-to-avatar <a href='https://huggingface.co/spaces/lambdalabs/text-to-avatar'>https://huggingface.co/spaces/lambdalabs/text-to-avatar</a></li> 
       <li>generative-music-visualizer <a href='https://huggingface.co/spaces/lambdalabs/generative-music-visualizer'>https://huggingface.co/spaces/lambdalabs/generative-music-visualizer</a></li> 
       <li>text-to-pokemon <a href='https://huggingface.co/spaces/lambdalabs/text-to-pokemon'>https://huggingface.co/spaces/lambdalabs/text-to-pokemon</a></li> 
       <li>image-mixer-demo <a href='https://huggingface.co/spaces/lambdalabs/image-mixer-demo'>https://huggingface.co/spaces/lambdalabs/image-mixer-demo</a></li> 
       <li>Stable Diffusion <a href='https://huggingface.co/blog/stable_diffusion'>https://huggingface.co/blog/stable_diffusion</a></li> 
       <li>CoDi: Any-to-Any Generation via Composable Diffusion <a href='https://codi-gen.github.io'>https://codi-gen.github.io</a></li> 
       <li>Imagen - Photorealistic Text-to-Image Diffusion Models <a href='https://imagen.research.google'>https://imagen.research.google</a></li> 
       <li>Pathways Autoregressive Text-to-Image model (Parti) <a href='https://sites.research.google/parti'>https://sites.research.google/parti</a></li> 
       <li>Muse: Text-To-Image Generation via Masked Generative Transformers <a href='https://muse-model.github.io'>https://muse-model.github.io</a></li> 
       <li>CLIP: clip-retrieval<a href='https://rom1504.github.io/clip-retrieval'>https://rom1504.github.io/clip-retrieval</a></li> 
       
       </ul>
       </p>
    """

    SECTION_FOOTER = """
       <p>Note: Only PNG is supported here, as of now</p>
       <p>Visit <a href='https://ai.amitpuri.com'>https://ai.amitpuri.com</a></p>
    """
    DISCLAIMER = """MIT License
    
    Copyright (c) 2023 Amit Puri
    

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    
    furnished to do so, subject to the following conditions:
    

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    
    """
    FOOTER = """<div class="footer">
                        <p>by <a href="https://www.amitpuri.com" style="text-decoration: underline;" target="_blank">Amit Puri</a></p>

                </div>            
            """
    
    
    AWESOME_CHATGPT_PROMPTS = """
    Credits 🧠 Awesome ChatGPT Prompts <a href='https://github.com/f/awesome-chatgpt-prompts'>https://github.com/f/awesome-chatgpt-prompts</a>
    """
    PRODUCT_DEFINITION_INFO = "<p>Prompt builder, <br><br> Step 1 - Select a fact sheet, <br><br> Step 2 - Select a task and <br><br> Step 3 - Select a question to build it <br><br> Step 4 - Click Ask ChatGPT</p>"
    
    PRODUCT_DEFINITION = "<p>Define a product by prompt, picturize it, get variations, save it with a keyword for later retrieval. Credits <a href='https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers'>https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers</a></p>"

    
    LABEL_GPT_CELEB_SCREEN = "Select, Describe, Generate AI Image, Upload and Save"

    NO_API_KEY_ERROR = "Review Configuration tab for keys/settings, OPENAI_API_KEY is missing or No input"

    NO_GOOGLE_PALM_AI_API_KEY_ERROR = "Review Configuration tab for keys/settings, PaLM Generative AI API Key (or GOOGLE_PALM_AI_API_KEY) is missing or No input"

    ENTER_A_PROMPT_IMAGE = "Please a prompt for image"
    
    PDF_OUTPUT_INFO = "PDF summarize Output info"
    
    TRANSCRIBE_OUTPUT_INFO = "Transcribe and summarize Output info"
    
    NO_API_KEY_ERROR_INVALID = "Review Configuration tab for keys/settings, OPENAI_API_KEY is invalid."
    
    NO_RAPIDAPI_KEY_ERROR = "Review Configuration tab for keys/settings, RAPIDAPI_KEY is missing or No input"
    
    TASK_EXPLANATION_EXAMPLES = ["""Your task is to help a marketing team create a description for a retail website of a product based on a technical fact sheet.
           
                                Write a product description based on the information provided in the technical specifications delimited by triple backticks."""]

    
    PRODUCT_DEF_QUESTION_EXAMPLES = ["Limit answer to 50 words", 
                                 "Limit answer to 100 words", 
                                 "Write the answer in bullet points",
                                 "Write the answer in 2/3 sentences",
                                 "Write the answer in one line TLDR with the fewest words"
                                ]

    ARTICLE_LINKS_EXAMPLES = ["https://time.com/6266679/musk-ai-open-letter", 
                              "https://futureoflife.org/open-letter/ai-open-letter",
                              "https://github.com/openai/CLIP",
                              "https://arxiv.org/abs/2103.00020",
                              "https://arxiv.org/abs/2302.14045v2",
                              "https://arxiv.org/abs/2304.04487",
                              "https://arxiv.org/abs/2212.09611",
                              "http://arxiv.org/abs/2305.02897",
                              "https://arxiv.org/abs/2305.00050",
                              "https://arxiv.org/abs/2304.14473",
                              "https://arxiv.org/abs/1607.06450",
                              "https://arxiv.org/abs/1706.03762",
                              "https://spacy.io/usage/spacy-101",
                              "https://developers.google.com/machine-learning/gan/gan_structure",
                              "https://thegradient.pub/nlp-imagenet",
                              "https://arxiv.org/abs/2102.12092",
                              "https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html",
                              "https://lukesalamone.github.io/posts/what-is-temperature",
                              "https://langchain.com/features.html",
                              "https://arxiv.org/abs/2010.11929",
                              "https://developers.google.com/machine-learning/gan/generative"]




    KEYWORD_EXAMPLES = sorted(["Stable Diffusion", "Zero-shot classification", "Generative AI based Apps ", 
                    "Foundation Capital FMOps ", "Foundational models AI", "Prompt Engineering", "Generative AI", 
        		    "Hyperparameter optimization","Embeddings Search","Convolutional Neural Network","Recurrent neural network",
                    "XGBoost Grid Search", "Random Search" , "Bayesian Optimization", "NLP", "GPT", "Vector Database",
                    "OpenAI embeddings","ChatGPT","Python LangChain LLM", "Popular LLM models", "Hugging Face Transformer",
                    "Confusion Matrix", "Feature Vector", "Gradient Accumulation","Loss Functions","Cross Entropy",
                    "Root Mean Square Error", "Cosine similarity", "Euclidean distance","Dot product similarity",
                    "Machine Learning","Artificial Intelligence","Deep Learning", "Neural Networks", "Data Science",
                    "Supervised Learning","Unsupervised Learning","Reinforcement Learning", "Natural Language Processing", 
                    "Data Mining", "Feature Extraction", "Dimensionality Reduction", "Ensemble Learning", "Transfer Learning",
                    "Decision Trees","Support Vector Machines", "Clustering","Regression", "Computer Vision", "Big Data",                   
                    "Language Models","Transformer","BERT","OpenAI","Text Generation","Text Classification",
                    "Chatbots","Summarization","Question Answering","Named Entity Recognition","Sentiment Analysis",
                    "Pretraining","Finetuning","Contextual Embeddings","Attention Mechanism","Reinforcement learning",
                    "Pinecone, a fully managed vector database", "Weaviate, an open-source vector search engine",
                    "Redis as a vector database","Qdrant, a vector search engine", "Milvus", "Embedding-based search",
                    "Chroma, an open-source embeddings store","Typesense, fast open source vector search", "Low-code No-code",
                    "Zilliz, data infrastructure, powered by Milvus", "Lexical-based search","Graph-based search"                            

                   ])


    prompt_character = PromptTemplate(
    input_variables=["character_name","program_name"],
    template="What is the name of the actor acted as {character_name} in {program_name}, answer without any explanation and return only the actor's name?")

    prompt_bond_girl = PromptTemplate(
    input_variables=["movie_name"],
    template="Who was Bond girl co-star in {movie_name}? answer without any explanation and return only the actor's name?")


    CELEB_SEARCH_QUESTIONS_EXAMPLES = sorted([prompt_character.format(character_name="Princess Devasena", program_name="Movie Baahubali"),
                        prompt_character.format(character_name="Mahendra/Amerendra Baahubali", program_name="Movie Baahubali"),                        
                        prompt_character.format(character_name="Kattappa", program_name="Movie Baahubali"),
                        prompt_character.format(character_name="Sivagami Devi", program_name="Movie Baahubali"),                                                
                        prompt_character.format(character_name="James Bond", program_name="Casino Royale"),
                        prompt_character.format(character_name="James Bond", program_name="Die Another Day"),
                        prompt_character.format(character_name="James Bond", program_name="Never Say Never Again"),
                        prompt_character.format(character_name="James Bond", program_name="Spectre"),
                        prompt_character.format(character_name="James Bond", program_name="Tomorrow Never Dies"),
                        prompt_character.format(character_name="James Bond", program_name="The World Is Not Enough"),
                        prompt_character.format(character_name="James Bond", program_name="Goldfinger"),
                        prompt_character.format(character_name="James Bond", program_name="Octopussy"),
                        prompt_character.format(character_name="James Bond", program_name="Diamonds Are Forever"),
                        prompt_character.format(character_name="James Bond", program_name="Licence to Kill"), 
                        prompt_character.format(character_name="Patrick Jane", program_name="The Mentalist"),
                        prompt_character.format(character_name="Raymond Reddington", program_name="The Blacklist"),
                        prompt_character.format(character_name="Tom Kirkman", program_name="Designated Survivor"),  
                        prompt_character.format(character_name="Carrie Wells", program_name="Unforgettable"),
                        prompt_character.format(character_name="Bhallaladeva, the king of Mahishmati", program_name="Movie Baahubali"),
                        prompt_character.format(character_name="Avanthika, a skilled warrior and a fighter of the rebel group", program_name="Movie Baahubali"),
                        prompt_bond_girl.format(movie_name="Casino Royale"),
                        prompt_bond_girl.format(movie_name="GoldenEye"),
                        prompt_bond_girl.format(movie_name="Spectre"),
                        prompt_bond_girl.format(movie_name="Tomorrow Never Dies"),
                        prompt_bond_girl.format(movie_name="Goldfinger"),
                        prompt_bond_girl.format(movie_name="No Time to Die"),
                        prompt_bond_girl.format(movie_name="Octopussy"),
                        prompt_bond_girl.format(movie_name="The World Is Not Enough"),
                        prompt_bond_girl.format(movie_name="Diamonds Are Forever"),
                        prompt_bond_girl.format(movie_name="Licence to Kill"),                          	
		                prompt_bond_girl.format(movie_name="Die Another Day")])


    coolest_midjourney_prompts =  [ "While traveling through a dense forest, you stumble upon an ancient, overgrown path veering off from the main trail. Do you dare to explore its mysteries?",
 "As you sail across the open sea, a sudden storm engulfs your ship, throwing you off course. You find yourself stranded on a remote, uncharted island. What secrets does it hold?",
 "During your intergalactic voyage, your spacecraft malfunctions and crash-lands on an unknown planet. As you step out, you discover a civilization unlike anything you've ever seen. How will you communicate with its inhabitants?",
 "While trekking through a barren desert, you uncover an entrance to an underground labyrinth. Its walls are adorned with ancient symbols and clues. Will you risk getting lost in its depths to unravel its enigmatic riddles?",
 "On your quest to summit a towering mountain, you encounter a legendary creature said to possess the power to grant a single wish to those who prove their worth. What challenges must you overcome to face this majestic being?",
 "As you journey through a bustling futuristic city, you stumble upon a hidden resistance group fighting against a tyrannical regime. Will you join their cause and help bring about change, or stay on the sidelines?",
 "While traversing a parallel dimension, you come across a mirror that shows you glimpses of your own future. How will these glimpses affect your choices and the outcome of your journey?",
 "During a long train ride, you strike up a conversation with a mysterious stranger who claims to have the ability to travel through time. Will you believe their incredible stories and join them on a mind-bending adventure?",
 "As you explore an ancient, abandoned ruin, you accidentally activate a long-lost artifact, transporting you to a bygone era. Can you find a way back home while navigating the perils of a time unknown to you?",
 "While backpacking through a remote wilderness, you stumble upon a hidden tribe that has managed to preserve ancient traditions and knowledge. Will you earn their trust and gain access to their wisdom?" ,
"Visualize the concept of thought leadership in a futuristic cityscape.",
"Illustrate the power of collaboration in a forest ecosystem.",
"Depict a tree growing from a seed to a mighty oak to symbolize a growth mindset.",
"Create a humorous scene of robots having a stressful day at work.",
"Show a sunrise over a mountain range, symbolizing new beginnings and opportunities.",
"Illustrate the concept of resilience in a desert oasis.",
"Depict a symphony orchestra as a metaphor for harmonious teamwork.",
"Visualize the journey of a single drop of water contributing to a mighty river, symbolizing individual contribution to collective success.",
"Create an image of a lighthouse standing firm amidst a storm, symbolizing guidance and steadfastness.",
"Illustrate a garden blooming in all seasons, symbolizing adaptability and continuous growth.",
"Show a humorous scene of animals conducting a business meeting.",
"Depict a chess game where all pieces are working together, symbolizing strategic collaboration.",
"Visualize a bridge connecting two cliffs, symbolizing problem-solving and connection.",
"Create a scene of a runner passing the baton in a relay race, symbolizing trust and teamwork.",
"Illustrate a vibrant coral reef, symbolizing diversity and symbiotic relationships.",
"Show a humorous scene of birds having a singing competition.",
"Depict a tree with deep roots and wide branches, symbolizing strength and growth.",
"Visualize a mountain climber reaching the peak, symbolizing achievement and ambition.",
"Create a scene of a spaceship exploring unknown galaxies, symbolizing curiosity and innovation.",
"Illustrate a cityscape where nature and technology coexist harmoniously.",
"Show a humorous scene of a robot trying to paint a masterpiece.",
"Depict a phoenix rising from the ashes, symbolizing resilience and rebirth.",
"Visualize a person planting seeds, symbolizing patience and investment in the future.",
"Create a scene of a tightrope walker, symbolizing balance and risk-taking.",
"Illustrate a network of interconnected gears, symbolizing synergy and cooperation.",
"Show a humorous scene of a group of animals having a picnic.",
"Depict a person standing at a crossroads, symbolizing decision-making and strategic planning.",
"Visualize a person building a sandcastle, symbolizing creativity and imagination.",
"Create a scene of a hot air balloon rising above the clouds, symbolizing aspiration and freedom.",
"Illustrate a group of diverse individuals holding hands, symbolizing unity and inclusivity.",
"Show a humorous scene of a robot trying to cook a gourmet meal.",
"Depict a person climbing a staircase made of books, symbolizing knowledge and progress.",
"Visualize a person standing at the edge of a cliff, looking at the horizon, symbolizing vision and foresight.",
"Create a scene of a person navigating through a maze, symbolizing problem-solving and determination.",
"Illustrate a group of people building a bridge, symbolizing collaboration and teamwork.",
"Show a humorous scene of a group of animals playing a game of soccer.",
"Depict a person planting a tree, symbolizing investment in the future and environmental consciousness.",
"Visualize a person painting a canvas, symbolizing creativity and self-expression.",
"Create a scene of a person reaching for a star, symbolizing ambition and aspiration.",
"Illustrate a group of people holding different pieces of a puzzle, symbolizing teamwork and collaboration.",
"Show a humorous scene of a robot trying to dance.",
"Depict a person standing at multiple doors, symbolizing choices and opportunities.",
"Visualize a person holding a light in the darkness, symbolizing hope and guidance.",
"Create a scene of a person sailing in a storm, symbolizing resilience and courage.",
"Illustrate a group of people from different cultures holding hands, symbolizing unity in diversity.",
"Show a humorous scene of a group of animals having a music concert.",
"Depict a person standing on top of a mountain, looking at the horizon, symbolizing achievement and vision.",
"Visualize a person walking on a path that leads to a bright future, symbolizing hope and progress.",
"Create a scene of a person building a house, symbolizing patience and hard work.",
"Illustrate a group of people carrying a giant globe, symbolizing shared responsibility and global cooperation."
]

    style_presets = ["enhance", "anime", "photographic", "digital-art", "comic-book", "fantasy-art", 
                    "line-art", "analog-film", "neon-punk", "isometric", 
                    "low-poly", "origami", "modeling-compound", "cinematic", 
                    "3d-model", "pixel-art", "tile-texture"]


    elevenlabs_voices = ["Rachel","Domi","Bella","Antoni","Elli","Josh","Arnold","Adam","Sam"]
    
    diffusion_models = ["prompthero/linkedin-diffusion", "prompthero/openjourney","runwayml/stable-diffusion-v1-5","CompVis/stable-diffusion-v1-4","stability.ai","dall-e"]

    audio_models = ["openai/whisper-1","speechbrain/speechbrain","assemblyai/assemblyai"]

    llm_api_options = ["OpenAI API","Azure OpenAI API","Google PaLM API"]

    llm_models = ["meta-llama/Llama-2-70b-chat-hf"]

    text2audio_medium = ["elevanlabs","microsoft/speecht5_tts","coqui/tts","speechbrain/tts-tacotron2-ljspeech"]

    text2image_medium = ["StabilityAI", "OpenAI API","Azure OpenAI API","Vertex AI Image Generation"]

    TEST_MESSAGE = "My favorite TV shows are The Mentalist, The Blacklist, Designated Survivor, and Unforgettable. What are ten series that I should watch next?"

    MONGODB_HTML = "Sign up here <a href='https://www.mongodb.com/cloud/atlas/register'>https://www.mongodb.com/cloud/atlas/register</a>"

    OPENAI_HTML = "Sign up for API Key here <a href='https://platform.openai.com'>https://platform.openai.com</a>"

    AZURE_OPENAI_HTML = "Apply for access to Azure OpenAI Service by completing the form at <a href='https://aka.ms/oai/access?azure-portal=true'>https://aka.ms/oai/access?azure-portal=true</a>"

    GOOGLE_PALMAPI_HTML = "Visit PaLM (Pathways Language Model) API <a href='https://developers.generativeai.google'>Google Generative AI </a>,  <a href='https://makersuite.google.com'>MakerSuite</a>, and <a href='https://developers.generativeai.google/develop/sample-apps'>https://developers.generativeai.google/develop/sample-apps</a>"

    CLOUDINARY_HTML = "Sign up here <a href='https://cloudinary.com'>https://cloudinary.com</a>"

    openai_models = ["gpt-4", "gpt-4-0613", "gpt-4-32k", "gpt-4-32k-0613", "gpt-3.5-turbo", 
                     "gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-16k-0613", "text-davinci-003", 
                     "text-davinci-002", "text-curie-001", "text-babbage-001", "text-ada-001"]

    google_palm_models = ["models/text-bison-001"]
    
    imagesize_text = "Select one, use download for image size from Image generation/variation Output tab"
    
    STABILITY_AI_HTML = "Sign up here <a href='https://platform.stability.ai'>https://platform.stability.ai</a> and learn how to write a prompt <a href='https://dreamstudio.ai/prompt-guide'>https://dreamstudio.ai/prompt-guide</a>"

    RAPIDAPI_HTML = "Sign up here <a href='https://rapidapi.com'>https://rapidapi.com</a>"


    RAPIDAPI_ARTICLE_HTML = "Article Extractor and Summarizer API on RapidAPI <a href='https://rapidapi.com/restyler/api/article-extractor-and-summarizer'>https://rapidapi.com/restyler/api/article-extractor-and-summarizer</a>"

    LANGCHAIN_TEXT = "Credit <a href='https://github.com/gkamradt/langchain-tutorials'>https://github.com/gkamradt/langchain-tutorials</a>"

    ASSEMBLY_AI_HTML = "Sign up AssemblyAI <a href='https://www.assemblyai.com'>https://www.assemblyai.com/dashboard/signup</a>"

    SPEECHBRAIN_HTML  = "Speechbrain <a href='https://huggingface.co/speechbrain'>https://huggingface.co/speechbrain</a>"

    ELEVENLABS_HTML = "Sign up Elevenlabs <a href='https://www.elevenlabs.io'>https://www.elevenlabs.io</a>"
    
    ELEVENLABS_TEST_MESSAGE ="AI as a tool that can augment and empower us, rather than compete or replace us."

    DIFFUSION_MODELS_HTML ="Diffusion Models from huggingface, Stability AI, OpenAI DALL-E"
    
    NO_ASSEMBLYAI_API_KEY_ERROR = "AssemblyAI API Key or ASSEMBLYAI_API_KEY env variable missing!"

    NO_STABILITYAI_API_KEY_ERROR = "StabilityAI API Key or STABILITYAI_API_KEY env variable missing!"

    TEXT_TO_VIDEO_HTML = """
            <li>The Task, Challenges and the Current State <a href='https://huggingface.co/blog/text-to-video'>https://huggingface.co/blog/text-to-video</a></li>
            <li><a href='https://imagen.research.google/video'>https://imagen.research.google/video</a></li>
            <li><a href='https://phenaki.video'>https://phenaki.video</a></li>
            <li><a href='https://huggingface.co/spaces/damo-vilab/modelscope-text-to-video-synthesis'>https://huggingface.co/spaces/damo-vilab/modelscope-text-to-video-synthesis</a></li>
            <li><a href='https://github.com/VideoCrafter/VideoCrafter'>https://github.com/VideoCrafter/VideoCrafter</a></li>
            <li><a href='https://github.com/THUDM/CogVideo'>https://github.com/THUDM/CogVideo</a></li>
            <li><a href='https://github.com/topics/text-to-video'>https://github.com/topics/text-to-video</a></li>
    """