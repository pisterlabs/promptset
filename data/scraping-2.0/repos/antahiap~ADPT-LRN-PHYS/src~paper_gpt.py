import ast
import json
from openai_api import OpenAIApi
from openai_multi_client import OpenAIMultiClient
from constants import PAPER_GPT_PATH


class PaperGPT():
    def __init__(self, content, paper_file_name):
        self.api = OpenAIApi("gpt-3.5-turbo-16k")
        self.content = content
        self.summary = None
        self.keywords = []
        self.keywords_explanations = {}
        self.info_pah = PAPER_GPT_PATH / f"{paper_file_name.stem}.json"
        self.keywords_db = database.Keywords()
        self.retrieve_info()

    def retrieve_info(self):
        if self.info_pah.exists():
            with open(self.info_pah, "r") as f:
                info = json.load(f)
            self.summary = info["summary"]
            self.keywords = info["keywords"]
            self.keywords_explanations = info["keywords_explanations"]
        else:
            self.generate_info()
            self.save_info()
    
    def generate_info(self):
        self.generate_summary()
        self.find_keywords()
        self.explain_keywords()

    def save_info(self):
        info = {
            "summary": self.summary,
            "keywords": self.keywords,
            "keywords_explanations": self.keywords_explanations,
        }
        with open(self.info_pah, "w") as f:
            json.dump(info, f, indent=4)

    def generate_summary(self):
        prompt = f"Answer using markdown. Explain this paper in every details using markdown: \n{self.content}\n"
        self.summary, _, _ = self.api.call_api(prompt)
        return self.summary
    
    def _parse_keywords(self, raw_keywords):
        try:
            keywords = ast.literal_eval(raw_keywords)
        except ValueError as e:
            print(f"Error: {e}\nFor raw_keywords: {raw_keywords}")
            keywords = []
        return keywords

    def _filter_keywords(self, keywords):
        return list({keyword for keyword, score in keywords if score >= 0.8}) # Set comprehension to only get unique keywords
    
    def find_keywords(self):
        for _ in range(3):
            prompt = f"What are the keywords of the text. Answer with their importance from 0 to 1. Answer in an array of tuples: \n{self.summary}\n"
            raw_keywords, _, _ = self.api.call_api(prompt, model="gpt-3.5-turbo")
            keywords = self._parse_keywords(raw_keywords)
            if keywords:
                self.keywords = self._filter_keywords(keywords)
                return self.keywords

    def add_explanation(self, result):
        keyword = result.metadata["keyword"]
        explanation = result.response['choices'][0]['message']['content']
        print(f"Result: {keyword}\n{explanation}")
        self.keywords_explanations[keyword] = explanation
    
    def explain_keywords(self):
        api = OpenAIMultiClient(endpoint="chats", data_template={"model": "gpt-3.5-turbo"})
        def make_requests():
            for index, keyword in enumerate(self.keywords):
                prompt = f"Answer using markdown. Explain '{keyword}' in using markdown: \n{self.summary}\n"
                print(f"Request {index} {keyword}")
                api.request(
                    data={"messages": [{"role": "user", "content": prompt}]},
                    metadata={'id': index, 'keyword': keyword},
                )
        api.run_request_function(make_requests)

        for result in api:
            self.add_explanation(result)
        return self.keywords_explanations

    def to_html(self, tooltip_class="tooltip", tooltip_text_class="tooltiptext"):
        html = self.summary
        placeholder = lambda keyword: f"__{keyword.upper()}__"
        for keyword, explanation in self.keywords_explanations.items(): # 2 Step to avoid replacing keywords inside of a tooltip
            html = html.replace(keyword, placeholder(keyword))
        for keyword, explanation in self.keywords_explanations.items():
            html = html.replace(placeholder(keyword), f'<span class="{tooltip_class}">{keyword}<span class="{tooltip_text_class}">{explanation}</span></span>')
        return html


if __name__ == "__main__":
    with open("data/paper.txt", "r") as f:
        content = f.read()
    print(len(content))
    paper = PaperGPT(content)
    paper.summary = "The paper introduces the Transformer , a new network architecture based solely on attention mechanisms for sequence transduction tasks. The Transformer removes the need for recurrent or convolutional layers typically used in encoder-decoder models, and instead connects the encoder and decoder with attention mechanisms. The model achieves superior results in terms of translation quality, parallelization, and training time on two machine translation tasks. It also generalizes well to other tasks, such as English constituency parsing. The authors provide detailed experimental results and variations of the model to evaluate the importance of different components."
    paper.keywords = ['Transformer', 'attention mechanisms', 'sequence transduction tasks', 'recurrent', 'convolutional layers', 'encoder-decoder models', 'translation quality', 'machine translation tasks', 'English constituency parsing', 'experimental results', 'importance of different components']
    paper.keywords_explanations = {'English constituency parsing': 'English constituency parsing is a task in natural language processing that involves identifying the syntactic structure of sentences and dividing them into constituent phrases. The paper mentions that the Transformer, a network architecture based on attention mechanisms, performs well in English constituency parsing along with other tasks.', 'attention mechanisms': 'Attention mechanisms are a key part of the Transformer network architecture. They allow the model to focus on different parts of the input sequence when generating an output sequence. This removes the need for recurrent or convolutional layers and improves translation quality, parallelization, and training time. The effectiveness of attention mechanisms is demonstrated through detailed experiments and variations of the model.', 'translation quality': "Translation quality refers to the level of accuracy and fluency in the translated output. In the context of the paper, the authors introduce a new network architecture called the Transformer, which achieves superior translation quality compared to traditional models. They achieve this by using attention mechanisms to connect the encoder and decoder, eliminating the need for recurrent or convolutional layers. The model's translation quality is evaluated through experimental results and comparisons with other models.", 'convolutional layers': 'Convolutional layers are a type of layer used in convolutional neural networks (CNNs) that apply filters or kernels to input data in order to extract features. These layers are commonly used in computer vision tasks for analyzing images or sequences of data. They allow the network to detect patterns and spatial relationships in the data by convolving the input with different filters and pooling the results. This helps in capturing local patterns and learning hierarchical representations of the input data.', 'experimental results': 'Experimental results refer to the outcomes obtained by conducting experiments to evaluate the performance and effectiveness of a particular model or approach. In the context of the paper explaining the Transformer network architecture, the experimental results would include the findings and measurements derived from testing the model on various tasks, such as machine translation and English parsing. These results demonstrate the superiority of the Transformer in terms of translation quality, parallelization, training time, and its ability to generalize to different tasks. The authors further provide variations of the model and conduct experiments to analyze and determine the significance of different components within the Transformer.', 'Transformer': 'In simple terms, the Transformer is a new type of network architecture that uses attention mechanisms to perform tasks involving sequences of data. It replaces the traditional recurrent or convolutional layers used in encoder-decoder models and connects the encoder and decoder with attention mechanisms instead. The Transformer has been shown to achieve excellent results in machine translation tasks, offering advantages in translation quality, parallelization, and training time. It also performs well on other tasks, like English parsing. The authors back up their claims with detailed experiments and variations of the model to assess the significance of various components.', 'recurrent': 'In the context of neural networks, "recurrent" refers to a specific type of layer or architecture that is commonly used for sequence transduction tasks. Recurrent layers are designed to process sequential data by allowing information to be passed from one step to the next, making them suitable for tasks like natural language processing or speech recognition. However, the Transformer network, introduced in the paper, does not rely on recurrent layers. Instead, it utilizes attention mechanisms to connect the encoder and decoder, which leads to improved translation quality, parallelization capabilities, and faster training time. The Transformer model also demonstrates good performance on various tasks beyond machine translation.', 'sequence transduction tasks': 'Sequence transduction tasks refer to tasks where a sequence of input elements is transformed into a sequence of output elements. The Transformer network architecture, introduced in the paper, is specifically designed for these tasks. Unlike traditional encoder-decoder models that use recurrent or convolutional layers, the Transformer uses attention mechanisms to connect the encoder and decoder. This eliminates the need for sequential processing and allows for better translation quality, faster parallelization, and reduced training time. The Transformer model also performs well on other tasks, such as English constituency parsing, and the authors provide extensive experimental results to analyze the impact of different components of the model.', 'machine translation tasks': 'Machine translation tasks refer to the process of automatically translating text or speech from one language to another using computational techniques. The Transformer is a network architecture that improves the quality, efficiency, and training time of machine translation. It replaces traditional recurrent or convolutional layers in encoder-decoder models with attention mechanisms, which connect the encoder and decoder. The Transformer achieves superior translation quality, parallelization, and training time on two specific machine translation tasks. Furthermore, it can effectively handle other tasks, such as English constituency parsing. The authors of the paper provide comprehensive experimental results and variations of the model to assess the significance of various components.', 'encoder-decoder models': 'Encoder-decoder models are a type of neural network architecture commonly used in sequence transduction tasks, such as machine translation. These models consist of an encoder and a decoder. The encoder processes the input sequence and converts it into a fixed-length representation, while the decoder generates the output sequence based on this representation. In traditional encoder-decoder models, recurrent or convolutional layers are used for encoding and decoding. However, the Transformer architecture introduced in the paper removes the need for these layers and instead uses attention mechanisms to connect the encoder and decoder. This results in improved translation quality, faster training time, and better parallelization. The Transformer model can also be applied to other tasks and exhibits good generalization. The paper provides detailed experimental results and explores various variations of the model to analyze the importance of different components.', 'importance of different components': 'The different components in the Transformer network architecture play a crucial role in achieving superior results in sequence transduction tasks. By using attention mechanisms instead of recurrent or convolutional layers, the need for complex computations is reduced, resulting in improved translation quality, parallelization, and reduced training time. The Transformer generalizes well to various tasks, highlighting the importance of these components in enhancing performance. The authors assess the significance of each component through detailed experimental results and variations of the model.'}
    # res = paper.find_keywords()
    # res = paper.explain_keywords()
    res = paper.to_html()
    print(res)
