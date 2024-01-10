import openai 
import os
import json
import argparse
import yaml
from tqdm import tqdm
from sklearn.metrics import accuracy_score, hamming_loss, precision_score, recall_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer


openai.organization = os.environ["OPENAI_ORGANIZATION"]
openai.api_key = os.getenv("OPENAI_API_KEY") 

class OpenAiTextClassificationPropagandaInference:
    def __init__(self, config_path):
        self.config_path = config_path
        self.load_config()
        # self.load_data()

    def load_config(self):    
        with open(self.config_path, 'r') as stream:
            self.model_config = yaml.safe_load(stream)

    def prompt_gen(self, input_text):
        prompt_instruction = f"""You are a multi-label text classifier indetifying 14 propaganda techniques within news paper articles. These are the 14 propaganda techniques you classify with definitions and examples:
        Loaded_Language - Uses specific phrases and words that carry strong emotional impact to affect the audience, e.g. 'a lone lawmaker’s childish shouting.'
        Name_Calling,Labeling - Gives a label to the object of the propaganda campaign as either the audience hates or loves, e.g. 'Bush the Lesser.'
        Repetition -  Repeats the message over and over in the article so that the audience will accept it, e.g. 'Our great leader is the epitome of wisdom. Their decisions are always wise and just.'
        Exaggeration,Minimisation - Either representing something in an excessive manner or making something seem less important than it actually is, e.g. 'I was not fighting with her; we were just playing.'
        Appeal_to_fear-prejudice - Builds support for an idea by instilling anxiety and/or panic in the audience towards an alternative, e.g. 'stop those refugees; they are terrorists.'
        Flag-Waving; Playing on strong national feeling (or with respect to a group, e.g., race, gender, political preference) to justify or promote an action or idea, e.g. 'entering this war will make us have a better future in our country.'
        Causal_Oversimplification -  Assumes a single reason for an issue when there are multiple causes, e.g. 'If France had not declared war on Germany, World War II would have never happened.'
        Appeal_to_Authority - Supposes that a claim is true because a valid authority or expert on the issue supports it, 'The World Health Organisation stated, the new medicine is the most effective treatment for the disease.'
        Slogans - A brief and striking phrase that contains labeling and stereotyping, e.g.  “Make America great again!”
        Thought-terminating_Cliches -  Words or phrases that discourage critical thought and useful discussion about a given topic, e.g. “it is what it is”
        Whataboutism,Straw_Men,Red_Herring - Attempts to discredit an opponent’s position by charging them with hypocrisy without directly disproving their argument, e.g. 'They want to preserve the FBI’s reputation.'
        Black-and-White_Fallacy -  Gives two alternative options as the only possibilities, when actually more options exist, e.g. 'You must be a Republican or Democrat'
        Bandwagon,Reductio_ad_hitlerum - Justify actions or ideas because everyone else is doing it, or reject them because it's favored by groups despised by the target audience, e.g. “Would you vote for Clinton as president? 57% say yes.”
        Doubt - Questioning the credibility of someone or something, e.g. 'Is he ready to be the Mayor?'
        """
        if self.model_config['prompt_type'] == 'base':
            prompt_base = f"""
            For the given article please state which of the 14 propaganda techniques are present. If no propaganda technique was identified return "no propaganda detected". An example output would list the propaganda techniques with each technique in a new line, e.g.:
            Loaded_Language
            Thought-terminating_Cliches
            Repetition
            Here is the article:
            """
            prompt = f'{prompt_instruction}  {prompt_base} <{input_text}>'
        if self.model_config['prompt_type'] == 'chain_of_thought':
            prompt_chain_of_thought =  f"""
            For the given article please state which of the 14 propaganda techniques are present and give an explanation to why the technique is present in the article. If no propaganda technique was identified return "no propaganda detected". An example output would list the propaganda techniques with each technique in a new line, e.g.:
            Loaded_Language - Your explanation why this technique is present in the article.
            Thought-terminating_Cliches - Your explanation why this technique is present in the article.
            Repetition - Your explanation why this technique is present in the article.
            Here is the article:
            """
            prompt = f'{prompt_instruction} {prompt_chain_of_thought} <{input_text}>'
            # prompt = f'<{input_text}>'
        if self.model_config['instruction'] == False:
            prompt = f'<{input_text}>'
        return prompt
    
    def inference(self, prompt, model_name):
        if model_name in ['gpt-3.5-turbo', 'gpt-4']:
            completion = openai.ChatCompletion.create(
            model = model_name,
            messages = [
                {'role': 'assistant', 'content': f"""{prompt}"""},
            ],
            n = 1,
            stop = None,
            temperature=0.0, # set to 0 to get deterministic results
            timeout=100
            )
            return completion['choices'][0]['message']['content']
        else:
            completion = openai.Completion.create(
            engine=model_name,
            prompt=prompt,
            temperature=0.0,  # set to 0 to get deterministic results
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,

            )
            return completion['choices'][0]['text']
        
    
    def save_results(self):
        # create folder for results 
        if not os.path.exists(os.path.join('src/text_classification/gpt/results/', self.model_config['model_name'] + "-" + self.model_config['prompt_type'], 'dev-articles-labels-pred')):
            os.makedirs(os.path.join('src/text_classification/gpt/results/', self.model_config['model_name'] + "-" + self.model_config['prompt_type'], 'dev-articles-labels-pred'))
        # run inference on dev set and save results
        articles = sorted([filename for filename in os.listdir(self.model_config['val_data_path']) if filename.endswith('.txt')])
        print("run inference on dev set and save results")
        for article in tqdm(articles):
            full_article_path = os.path.join(self.model_config['val_data_path'], article)
            if os.path.exists(os.path.join('src/text_classification/gpt/results/', self.model_config['model_name'] + "-" + self.model_config['prompt_type'], 'dev-articles-labels-pred', article)): # check if prediction already exists
               continue
            with open(full_article_path, 'r', encoding='utf-8') as f:
                input_text = f.read()
            print(article)
            prompt = self.prompt_gen(input_text)
            model_output = self.inference(prompt, self.model_config['model_name'])
            if model_output == '':
                print("WARNING: model output is empty")
                model_output = 'no propaganda detected'
            with open(os.path.join('src/text_classification/gpt/results/', self.model_config['model_name'] + "-" + self.model_config['prompt_type'], 'dev-articles-labels-pred', article), 'w', encoding="utf-8") as f:
                f.write(model_output)
            

        

    def calculate_metrics(self):
        true_labels = []
        pred_labels = []

        val_label_path = self.model_config['val_label_path']
        pred_label_path = os.path.join('src/text_classification/gpt/results/', self.model_config['model_name'] + "-" + self.model_config['prompt_type'], 'dev-articles-labels-pred')

        filenames = sorted([filename for filename in os.listdir(val_label_path) if filename.endswith('.txt')])
        for filename in filenames:
            true_label_file = []
            file_path = os.path.join(val_label_path, filename)
            if os.stat(file_path).st_size == 0:
                true_labels.append(true_label_file) 
            else:
                with open(file_path, 'r', encoding="utf-8") as file:
                    for line in file:
                        line = line.strip()
                        true_label_file.append(line)
                true_labels.append(true_label_file)
            pred_label_file = []
            allowed_labels = ['Appeal_to_Authority', 'Appeal_to_fear-prejudice', 'Bandwagon,Reductio_ad_hitlerum', 'Black-and-White_Fallacy', 'Causal_Oversimplification', 'Doubt', 'Exaggeration,Minimisation', 'Flag-Waving', 'Loaded_Language', 'Name_Calling,Labeling', 'Repetition', 'Slogans', 'Thought-terminating_Cliches', 'Whataboutism,Straw_Men,Red_Herring']
            with open(os.path.join(pred_label_path, filename), 'r', encoding="utf-8") as file:
                if os.stat(os.path.join(pred_label_path, filename)).st_size == 0:
                    pred_labels.append(pred_label_file)
                else:
                    if self.model_config['prompt_type'] in ['base', 'noinstruction']:
                        for line in file:
                            line = line.strip()
                            if line in allowed_labels:
                                pred_label_file.append(line)
                    elif self.model_config['prompt_type'] == 'chain_of_thought':
                        for line in file:
                            line = line.strip()
                            label = line.split('-', 1)[0].strip()
                            if label in allowed_labels:
                                pred_label_file.append(label)
                    pred_labels.append(pred_label_file)


        # Convert labels to binary representation using MultiLabelBinarizer
        mlb = MultiLabelBinarizer()
        true_labels_binary = mlb.fit_transform(true_labels)
        predicted_labels_binary = mlb.transform(pred_labels)

        # Define the output file path
        metric_file_path = os.path.join('src/text_classification/gpt/results/', self.model_config['model_name'] + "-" + self.model_config['prompt_type'], 'metrics.txt')

        # Open the file in write mode
        with open(metric_file_path, "w") as file:
            # Accuracy
            accuracy = accuracy_score(true_labels_binary, predicted_labels_binary)
            file.write(f"Accuracy: {accuracy}\n")

            # Hamming Loss
            hamming_loss_value = hamming_loss(true_labels_binary, predicted_labels_binary)
            file.write(f"Hamming Loss: {hamming_loss_value}\n")

            # Precision, Recall, and F1-Score (Micro-Averaged)
            precision = precision_score(true_labels_binary, predicted_labels_binary, average='micro')
            recall = recall_score(true_labels_binary, predicted_labels_binary, average='micro')
            f1 = f1_score(true_labels_binary, predicted_labels_binary, average='micro')
            file.write(f"Micro-Averaged Precision: {precision}\n")
            file.write(f"Micro-Averaged Recall: {recall}\n")
            file.write(f"Micro-Averaged F1-Score: {f1}\n")

            # Subset Accuracy
            subset_accuracy = accuracy_score(true_labels_binary, predicted_labels_binary, normalize=True)
            file.write(f"Subset Accuracy: {subset_accuracy}\n")

            # Macro-Averaged Precision, Recall, and F1-Score
            macro_precision = precision_score(true_labels_binary, predicted_labels_binary, average='macro')
            macro_recall = recall_score(true_labels_binary, predicted_labels_binary, average='macro')
            macro_f1 = f1_score(true_labels_binary, predicted_labels_binary, average='macro')
            file.write(f"Macro-Averaged Precision: {macro_precision}\n")
            file.write(f"Macro-Averaged Recall: {macro_recall}\n")
            file.write(f"Macro-Averaged F1-Score: {macro_f1}\n")

            # F1-Score per class
            class_f1_scores = f1_score(true_labels_binary, predicted_labels_binary, average=None)
            file.write("F1-Score per Class:\n")
            for label, f1_score_class in zip(mlb.classes_, class_f1_scores):
                file.write(f"{label}: {f1_score_class}\n")
        print("Evaluation metrics written to the file:", metric_file_path)

    def run_all(self):
        self.save_results()
        self.calculate_metrics()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', help="Specify the path to model config yaml file", required=True)
    args = parser.parse_args()
    OpenAiTextClassificationPropagandaInference_ = OpenAiTextClassificationPropagandaInference(args.config_path)
    OpenAiTextClassificationPropagandaInference_.run_all()
#openai api fine_tunes.create -t test.jsonl -m ada --suffix "custom model name"

if __name__ == '__main__':
    main()