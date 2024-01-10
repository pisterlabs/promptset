import torch
import nltk
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import sys

import bs4 as bs  # beautifulsource4
import re
import logging
import os
import openai

nltk.download('punkt')
log = logging.getLogger(__name__)
api_key = os.environ.get("OPENAI_API_KEY")

PERCENTAGE = {"short": 30,
              "medium": 60,
              "long": 90
              }


def percentage(percent, whole):
    return (percent * whole) / 100.0


def number_of_words(article_text):
    # log.info(article_text)
    word_count = len(article_text.split(" "))
    return word_count


def summary_length(number, count):
    return round(percentage(number, count))


def nest_sentences(document):
    nested = []
    sent = []
    length = 0
    for sentence in nltk.sent_tokenize(document):
        length += len(sentence)
        if length < 1024:
            sent.append(sentence)
        else:
            nested.append(sent)
            sent = [sentence]
            length = len(sentence)

    if sent:
        nested.append(sent)

    return nested


def preprocess(url):
    # req = Request(url, headers={'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,'
    #                                       'image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    #                             'Accept-Encoding': 'gzip, deflate',
    #                             'Accept-Language': 'en-US,en;q=0.9',
    #                             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, '
    #                                           'like Gecko) Chrome/103.0.0.0 Safari/537.36'})
    header = {'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,'
                        'image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
              'Accept-Encoding': 'gzip, deflate',
              'Accept-Language': 'en-US,en;q=0.9',
              'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, '
                            'like Gecko) Chrome/103.0.0.0 Safari/537.36'}
    try:
        session = requests.Session()
        retry = Retry(connect=3, backoff_factor=0.5)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)

        article = session.get(url, headers=header).text
        # scraped_data = urlopen(req, timeout=3)
        # article = scraped_data.read()

        # parsed_article = bs.BeautifulSoup(article, 'lxml')
        #
        # paragraphs = parsed_article.find_all('p')
        #
        # article_text = ""
        #
        # for p in paragraphs:
        #     article_text += ' ' + p.text
        # formatted_article_text = re.sub(r'\n|\r', ' ', article_text)
        # formatted_article_text = re.sub(r' +', ' ', formatted_article_text)
        # formatted_article_text = formatted_article_text.strip()
        # return formatted_article_text
        return article

    except requests.ConnectionError as e:

        log.info("OOPS!! Connection Error. Make sure you are connected to Internet. Technical Details given below.\n")

        log.info(str(e))

    except requests.Timeout as e:

        log.info("OOPS!! Timeout Error")

        log.info(str(e))

    except requests.RequestException as e:

        log.info("OOPS!! General Error")

        log.info(str(e))

    except KeyboardInterrupt:

        log.info("Someone closed the program")


class SummarizerProcessor:
    def __init__(self, model: str = None):
        # log.info(model)
        # torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # log.info(torch_device)
        # if model is None:
        #     model = "t5"
        # self.modelName = model
        # # path to all the files that will be used for inference
        # self.path = f"./app/api/{model}/"
        # self.model_path = self.path + "pytorch_model.bin"
        # self.config_path = self.path + "config.json"
        #
        # # Selecting the correct model based on the passed madel input. Default t5
        # if model == "t5":
        #     self.config = T5Config.from_json_file(self.config_path)
        #     self.model = T5ForConditionalGeneration(self.config)
        #     self.tokenizer = T5Tokenizer.from_pretrained(self.path)
        #     self.model.eval()
        #     self.model.load_state_dict(torch.load(self.model_path, map_location=torch_device))
        # elif model == "google/pegasus-newsroom":
        #     self.config = PegasusConfig.from_json_file(self.config_path)
        #     # self.model = PegasusForConditionalGeneration(self.config)
        #     # self.tokenizer = PegasusTokenizer.from_pretrained(self.path)
        #     self.model = PegasusForConditionalGeneration.from_pretrained(model).to(torch_device)
        #     self.tokenizer = PegasusTokenizer.from_pretrained(model)
        # elif model == "facebook/bart-large-cnn":
        #     self.config = BartConfig.from_json_file(self.config_path)
        #     # self.model = PegasusForConditionalGeneration(self.config)
        #     # self.tokenizer = PegasusTokenizer.from_pretrained(self.path)
        #     self.model = BartForConditionalGeneration.from_pretrained(model).to(torch_device)
        #     self.tokenizer = BartTokenizer.from_pretrained(model)
        # else:
        #     raise Exception("This model is not supported")

        self.text = str()

    def generate_summary(self, nested_sentences, max_length):
        # logger.info("Inside inference before generate summary")
        # logger.info(self.model.get_input_embeddings())
        torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        summaries = []
        for nested in nested_sentences:
            input_tokenized = self.tokenizer.encode(' '.join(nested), truncation=True, return_tensors='pt')
            input_tokenized = input_tokenized.to(torch_device)
            summary_ids = self.model.to(torch_device).generate(input_tokenized,
                                                               length_penalty=3.0,
                                                               max_length=max_length)
            output = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
                      summary_ids]
            summaries.append(output)

        # logger.info("Inside inference after generate summary")

        summaries = [sentence for sublist in summaries for sentence in sublist]
        return summaries

    def generate_simple_summary(self, text):
        # logger.info("Inside inference before generate summary")
        # logger.info(self.model.get_input_embeddings())
        inputs = self.tokenizer([text], max_length=1024, return_tensors='pt')

        # Generate Summary
        summary_ids = self.model.generate(inputs['input_ids'], num_beams=4, max_length=5, early_stopping=True)
        # print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])
        # input_tokenized = self.tokenizer.encode(text, truncation=True, return_tensors='pt')
        # input_tokenized = input_tokenized.to(torch_device)
        # summary_ids = self.model.to(torch_device).generate(input_tokenized)
        output = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
                  summary_ids]

        return output

    async def inference(self, input_url: str, input_text: str, length: str):
        """
        Method to perform the inference
        :param input_text:
        :param text:
        :param length:
        :param input_url: Input url for the inference

        :return: correct category and confidence for that category
        """
        try:
            if input_url is not None:
                # self.text = preprocess(input_url)
                self.text = input_url
            else:
                self.text = input_text
            # log.info(api_key)
            # log.info(self.text)
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt="Please summarize the following article: " + self.text,
                max_tokens=200,
                n=1,
                stop=None,
                temperature=0.5,
            )
            summary = response["choices"][0]["text"]
            return summary

        # except Exception as e:
        except openai.OpenAIError as e:
            log.info("OpenAI API call failed with error:", e)
            # Handle exceptions that may occur while sending the GET request or the API request
            log.info("Error: Failed to fetch article or generate summary:", e)
            return "error"

    async def get_title(self, input_url: str, input_text: str, length: str):
        """
            Method to perform the inference
            :param input_text:
            :param text:
            :param length:
            :param input_url: Input url for the inference

            :return: correct category and confidence for that category
            """
        try:
            if input_url is not None:
                # self.text = preprocess(input_url)
                self.text = input_url
            else:
                self.text = input_text
            # log.info(api_key)
            # log.info(self.text)
            # Set up the prompt for the OpenAI API
            prompt = f"Please provide the title of the webpage at the following URL: {self.text}"

            # Send a request to the OpenAI API to generate a completion
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=100,
                n=1,
                stop=None,
                temperature=0.5,
            )

            # Extract the title from the OpenAI API response
            title = response.choices[0].text.strip()

            # Remove any newline or carriage return characters from the title
            title = re.sub('[\n\r]+', '', title)
            return title

        # except Exception as e:
        except openai.OpenAIError as e:
            log.info("OpenAI API call failed with error:", e)
            # Handle exceptions that may occur while sending the GET request or the API request
            log.info("Error: Failed to fetch article or generate summary:", e)
            return "error"
    # length_of_summary = PERCENTAGE[length]
    # max_length = 1000
    # if self.modelName == "google/pegasus-newsroom":
    #     batch = self.tokenizer(self.text, truncation=True, padding='longest', return_tensors="pt").to(torch_device)
    #     translated = self.model.generate(**batch)
    #     tgt_text = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
    #     # log.info(tgt_text)
    # elif self.modelName == "facebook/bart-large-cnn":
    #     nested = nest_sentences(self.text)
    #     summarized_text = self.generate_summary(nested, max_length)
    #     list_length = len(summarized_text)
    #     log.info(input_url)
    #     log.info(list_length)
    #     number_items = summary_length(list_length, length_of_summary)
    #     if number_items == 0:
    #         number_items = 1
    #     # nested_summ = nest_sentences(' '.join(summarized_text))
    #     # tgt_text_list = self.generate_summary(nested_summ,  max_length)
    #     index = 0
    #     tgt_text = ""
    #     while index < number_items:
    #         tgt_text += summarized_text[index]
    #         index += 1
    #     # tgt_text = self.generate_simple_summary(self.text)
    # return tgt_text
