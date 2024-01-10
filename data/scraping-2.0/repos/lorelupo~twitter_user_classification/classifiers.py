import openai
import backoff
import os
import re
import time
import collections 
from tqdm import tqdm
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from utils import setup_logging
from logging import getLogger, StreamHandler
logger = getLogger(__name__)
logger_backoff = getLogger('backoff').addHandler(StreamHandler())

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification

class LMClassifier:
    def __init__(
            self,
            labels_dict,
            label_dims,
            default_label,
            instruction,
            prompt_suffix,
            model_name,
            max_len_model,
            output_dir=None,
            log_to_file=True
            ):

        setup_logging(os.path.basename(__file__).split('.')[0], logger, output_dir if log_to_file else None)

        self.labels_dict = labels_dict
        # check the dimensionality of the labels:
        # dimensionality greater than 1 means dealing with
        # multiple classification tasks at a time
        self.label_dims = label_dims
        assert self.label_dims > 0, "Labels dimensions must be greater than 0."
        self.default_label = default_label
        
        # Define the instruction and ending ending string for prompt formulation
        # If instruction is a path to a file, read the file, else use the instruction as is
        self.instruction = open(instruction, 'r').read() if os.path.isfile(instruction) else instruction
        self.prompt_suffix = prompt_suffix.replace('\\n', '\n')

        self.max_len_model = max_len_model
        self.model_name = model_name

    def generate_predictions(self):
        raise NotImplementedError

    def range_robust_get_label(self, prediction, bounds):
        # more robust get label function that manages numbers in the returned text and assigns them to the correct range in case of number ranges
        # extract all two digit numbers or 0 from the prediction
        numbers = [int(n) for n in re.findall('\d{2}|[0]',prediction)]
        if len(numbers)==0:
            return self.labels_dict.get(self.default_label)
        if len(numbers)>0:
            if (numbers[-1]>bounds[-1][-1]) or (numbers[0]<bounds[0][0]):
                return self.labels_dict.get(self.default_label)
            elif len(numbers)==1:
                # check which list in bounds the number belongs to
                for i, bound in enumerate(bounds):
                    if numbers[0] in bound:
                        return self.labels_dict.get(list(self.labels_dict.keys())[i])
            elif len(numbers)>1:
                # just use the first 2 numbers
                # check the overlap of the range between numbers with bounds
                overlaps = [len(set(range(numbers[0],numbers[1])).intersection(set(bound))) for bound in bounds]
                return self.labels_dict.get(list(self.labels_dict.keys())[overlaps.index(max(overlaps))])


    def retrieve_predicted_labels(self, predictions, prompts=None, only_dim=None):

        # convert the predictions to lowercase
        predictions =  list(map(str.lower,predictions))

        # retrieve the labels that are contained in the predictions
        predicted_labels = []
        if self.label_dims == 1:
            # retrieve a single label for each prediction since a single classification task is performed at a time
            logger.info("Retrieving predictions...")
            for prediction in predictions:
                labels_in_prediction = [self.labels_dict.get(label) for label in self.labels_dict.keys() if label in prediction.split()]
                if len(labels_in_prediction) > 0:
                    predicted_labels.append(labels_in_prediction[0])
                else:
                    # first check if there is a range in all the labels
                    bounds = [[int(n) for n in key.split('-') if n.isnumeric()] for key in self.labels_dict.keys()]
                    if all(bounds): #if all labels have a number range
                        bounds = [list(range(b[0],b[1]+1)) for b in bounds]
                        predicted_labels.append(self.range_robust_get_label(prediction,bounds))
                    else:
                        predicted_labels.append(self.labels_dict.get(self.default_label))
            # Count the number of predictions of each type and print the result
            logger.info(collections.Counter(predicted_labels))
        else:
            # retrieve multiple labels for each prediction since multiple classification tasks are performed at a time
            logger.info(f"Retrieving predictions for {self.label_dims} dimensions...")
            for prediction in predictions:
                labels_in_prediction = []
                for dim in self.labels_dict.keys():
                    dim_label = []
                    for label in self.labels_dict[dim].keys():
                        if label in prediction:
                            dim_label.append(self.labels_dict[dim].get(label))   
                    dim_label = dim_label[0] if len(dim_label) > 0 else self.labels_dict[dim].get(self.default_label)
                    labels_in_prediction.append(dim_label)                                            
                predicted_labels.append(labels_in_prediction)
            # Count the number of predictions of each type and print the result
            logger.info(collections.Counter([",".join(labels) for labels in predicted_labels]))
        
        # Add the data to a DataFrame
        if self.label_dims == 1:
            df = pd.DataFrame({'prompt': prompts, 'prediction': predicted_labels}) if prompts else pd.DataFrame({'prediction': predicted_labels})
        elif self.label_dims > 1:
            if only_dim is not None:
                # retrieve only the predictions for a specific dimension
                logger.info(f"Retrieved predictions for dimension {only_dim}")
                df = pd.DataFrame({'prompt': prompts, 'prediction': pd.DataFrame(predicted_labels).to_numpy()[:,only_dim]}) if prompts else pd.DataFrame({'prediction': pd.DataFrame(predicted_labels).to_numpy()[:,only_dim]})
            else:
                logger.info("Retrieved predictions for all dimensions")
                df = pd.DataFrame(predicted_labels).fillna(self.default_label)
                # rename columns to prediction_n
                df.columns = [f"prediction_dim{i}" for i in range(1, len(df.columns)+1)]
                # add prompts to df
                if prompts:
                    df['prompt'] = prompts
        return df


class GPTClassifier(LMClassifier):
    def __init__(
            self,
            labels_dict,
            label_dims,
            default_label,
            instruction,
            prompt_suffix,
            model_name,
            max_len_model,
            gpt_system_role="You are a helpful assistant.",
            **kwargs,
            ):
        super().__init__(labels_dict, label_dims, default_label, instruction, prompt_suffix, model_name, max_len_model, **kwargs)  
        
        # set the average number of tokens per word in order to compute the max length of the input text
        self.avg_tokens_per_en_word = 4/3 # according to: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
        self.avg_tokens_per_nonen_word = 5 # adjust according to the language of the input text
        self.avg_tokens_per_word_avg = (self.avg_tokens_per_en_word + self.avg_tokens_per_nonen_word) / 2

        # if prompt is longer then max_len_model, we will remove words from the imput text
        # differently from HF models, where we have access to the tokenizer, here we work on full words
        len_instruction = len(self.instruction.split())
        len_output = len(self.prompt_suffix.split())
        self.max_len_input_text = int(
            (self.max_len_model - len_instruction*self.avg_tokens_per_en_word - len_output*self.avg_tokens_per_en_word) / self.avg_tokens_per_word_avg
            )

        # define the role of the system in the conversation
        self.system_role = gpt_system_role
        # load environment variables
        load_dotenv('.env')
        openai.api_key = os.getenv("OPENAI_API_KEY")

    @staticmethod
    @backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.APIError), max_tries=5)
    def completions_with_backoff(**kwargs):
        return openai.Completion.create(**kwargs) 

    @staticmethod
    @backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.APIError), max_tries=5)
    def chat_completions_with_backoff(**kwargs):
        return openai.ChatCompletion.create(**kwargs) 

    def generate_predictions(
            self,
            input_texts,
            sleep_after_step=0,
            ):

        prompts = []
        predictions = []

        # Generate a prompt and a prediction for each input text
        for i, input_text in enumerate(input_texts):
            # Create the prompt
            prompt = f'{self.instruction} {input_text} {self.prompt_suffix}'

            # if prompt is longer then max_len_model, remove words from the imput text
            len_prompt = int(len(prompt.split())*self.avg_tokens_per_word_avg)
            if len_prompt > self.max_len_model:
                # remove words from the input text
                input_text = input_text.split()
                input_text = input_text[:self.max_len_input_text]
                input_text = ' '.join(input_text)
                prompt = f'{self.instruction} {input_text} {self.prompt_suffix}'

                # print detailed info about the above operation
                logger.info(
                    f'Prompt n.{i} was too long, so we removed words from it. '
                    f'Approx original length: {len_prompt}; '
                    f'Approx new length: {int(len(prompt.split())*self.avg_tokens_per_word_avg)}'
                    )

            # log first prompt
            logger.info(prompt) if i == 0 else None

            # Print progress every 100 sentences
            if (i+1) % 20 == 0:
                logger.info(f"Processed {i+1} sentences")

            # Add the prompt to the list of prompts
            prompts.append(prompt)

            # call OpenAI's API to generate predictions
            try:
                # use chat completion for GPT3.5/4 models
                if self.model_name.startswith('gpt'):
                    gpt_out = self.chat_completions_with_backoff(
                        model=self.model_name,
                        messages=[
                            {"role": "system","content": self.system_role},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0,
                        max_tokens=15,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0
                    )
                    # Extract the predicted label from the output
                    predicted_label = gpt_out['choices'][0]['message']['content'].strip()

                    # Save predicted label to file, together with the index of the prompt
                    with open('raw_predictions_cache.txt', 'a') as f:
                        f.write(f'{i}\t{predicted_label}\n')

                    # Sleep in order to respect OpenAPI's rate limit
                    time.sleep(sleep_after_step)

                # use simple completion for GPT3 models (text-davinci, etc.)
                else:
                    gpt_out = self.completions_with_backoff(
                        model=self.model_name,
                        prompt=prompt,
                        temperature=0,
                        max_tokens=15,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0
                    )
                    # Extract the predicted label from the output
                    predicted_label = gpt_out['choices'][0]['text'].strip()

            # manage API errors
            except Exception as e:
                logger.error(f'Error in generating prediction for prompt n.{i}: {e}')
                # since the prediction was not generated, use the default label
                predicted_label = self.default_label
                logger.warning(f'Selected default label "{predicted_label}" for prompt n.{i}.')

            # Add the predicted label to the list of predictionss
            predictions.append(predicted_label)

        return prompts, predictions

class HFLMClassifier():
    def __init__(
            self,
            labels_dict,
            label_dims,
            default_label,
            instruction,
            prompt_suffix,
            model_name,
            max_len_model,
            output_dir=None,
            cache_dir=None,
            **kwargs,
            ):
                
        super().__init__(labels_dict, label_dims, default_label, instruction, prompt_suffix, model_name, max_len_model, output_dir, **kwargs)

        # Set device
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else None
        logger.info(f'Running on {self.device} device...')

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto", cache_dir=cache_dir)

    def generate_predictions(self, input_texts):

        # Encode the labels
        encoded_labels = self.tokenizer(list(self.labels_dict.keys()), padding=True, truncation=True, return_tensors="pt")['input_ids']
        logger.info(f'Encoded labels: \n{encoded_labels}')

        # Retrieve the tokens associated to encoded labels and print them
        # decoded_labels = tokenizer.batch_decode(encoded_labels)
        # print(f'Decoded labels: \n{decoded_labels}')
        max_len = max(encoded_labels.shape[1:])
        logger.info(f'Maximum length of the encoded labels: {max_len}')

        predictions = []
        prompts = []

        # Generate a prompt and a prediction for each input text
        for i, input_text in enumerate(input_texts):
            # Create the prompt
            prompt = f'{self.instruction} {input_text} {self.prompt_suffix}'

            # log first prompt
            logger.info(prompt) if i == 0 else None

            # Print progress every 100 sentences
            if (i+1) % 100 == 0:
                logger.info(f"Processed {i+1} sentences")

            # Add the prompt to the list of prompts
            prompts.append(prompt)

            # Activate inference mode
            torch.inference_mode(True)
            
            # Encode the prompt using the tokenizer and generate a prediction using the model
            with torch.no_grad():

                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

                # If inputs is longer then max_len_model, remove tokens from the encoded instruction
                len_inputs = inputs['input_ids'].shape[1]
                if len_inputs > self.max_len_model:
                    # get the number of tokens to remove from the encoded instruction
                    len_remove = len_inputs - self.max_len_model

                    # get the length of the output
                    len_output = self.tokenizer(self.prompt_suffix, return_tensors="pt")['input_ids'].shape[1] + 1 # +1 for the full stop token

                    # remove inputs tokens that come before the output in the encoded prompt
                    inputs['input_ids'] = torch.cat((inputs['input_ids'][:,:-len_remove-len_output], inputs['input_ids'][:,-len_output:]),dim=1)
                    inputs['attention_mask'] = torch.cat((inputs['attention_mask'][:,:-len_remove-len_output], inputs['attention_mask'][:,-len_output:]),dim=1)
                    
                    # print info about the truncation
                    logger.info(f'Original input text length: {len_inputs}. Input has been truncated to {self.max_len_model} tokens.')
                
                # Generate a prediction
                outputs = self.model.generate(**inputs, max_new_tokens=max_len) # or max_length=inputs['input_ids'].shape[1]+max_len
                predicted_label = self.tokenizer.decode(outputs[0].tolist(), skip_special_tokens=True)
                predictions.append(predicted_label)

            # Clear the cache after each iteration
            torch.cuda.empty_cache()

        return prompts, predictions
    
class HFClassifier:
    def __init__(
            self,
            model_name,
            max_len_model,
            batch_size=32,
            output_dir=None,
            cache_dir=None,
            log_to_file=True
            ):
        
        setup_logging(os.path.basename(__file__).split('.')[0], logger, output_dir if log_to_file else None)

        self.max_len_model = max_len_model
        self.model_name = model_name
        self.batch_size = batch_size
        self.dataloader = None
                
        # Set device
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else None
        logger.info(f'Running on {self.device} device...')

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype="auto", device_map="auto", cache_dir=cache_dir)

    @staticmethod
    def batch_tokenize(X_text, tokenizer, max_length=512, batch_size=64):

        # Dictionary to hold tokenized batches
        encodings = {}

        # Calculate the number of batches needed
        num_batches = len(X_text) // batch_size + int(len(X_text) % batch_size > 0)

        # Iterate over the data in batches
        for i in range(num_batches):
            batch_start = i * batch_size
            batch_end = min(len(X_text), (i + 1) * batch_size)

            # Tokenize the current batch of texts
            batch_encodings = tokenizer.batch_encode_plus(
                list(X_text[batch_start:batch_end]),
                padding='max_length',
                truncation=True,
                max_length=max_length
            )

            # Merge the batch tokenizations into the main dictionary
            for key, val in batch_encodings.items():
                if key not in encodings:
                    encodings[key] = []
                encodings[key].extend(val)

        return encodings

    def generate_predictions(self, input_texts):

        #TODO add input_ids when saving predictions to file, scores...

        logger.info(f'Tokenizing input texts...')
        encodings = HFClassifier.batch_tokenize(input_texts, self.tokenizer)

        # Set the model to evaluation mode
        self.model.eval()
        
        # store predicted probs
        predictions = []
        class_probs = []
        # Running inference on the model
        logger.info(f'Running model on batches...')
        with torch.no_grad():
            for i in tqdm(range(0, len(encodings['input_ids']), self.batch_size)):

                # Get the current batch and send it to GPU
                input_ids = torch.tensor(encodings['input_ids'][i:i+self.batch_size]).to(self.device)
                attention_mask = torch.tensor(encodings['attention_mask'][i:i+self.batch_size]).to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs['logits']

                # Get the predicted class labels
                predicted_labels = torch.argmax(logits, dim=1)
                predictions.extend(predicted_labels.cpu().numpy())

                # Convert logits to probabilities
                probabilities = F.softmax(logits, dim=1)
                class_probs.extend(probabilities.cpu().numpy().tolist())

        self.predictions = np.array(predictions)
        
        return self.predictions, np.array(class_probs)