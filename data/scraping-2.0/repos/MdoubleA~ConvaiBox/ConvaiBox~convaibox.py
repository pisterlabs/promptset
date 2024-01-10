from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, cached_path
import random
from os import path
import tarfile
from pathlib import Path
import interface
import torch


class ConvaiBox:
    # Constant from Hugging Faces' source code.
    HF_FINETUNED_MODEL = "https://s3.amazonaws.com/models.huggingface.co/transfer-learning-chatbot" \
                         "/gpt_personachat_cache.tar.gz"
    model_path = ".\\trained_model"

    # Set hyperparameters.
    no_sample = True  # Set to use greedy decoding instead of sampling
    max_out_utter = 20  # Maximum length of the output utterances
    min_out_utter = 1  # Minimum length of the output utterances
    seed = 0  # test what happens when using random number.
    temperature = 0.7  # Sampling softmax temperature
    top_k = 0  # Filter top-k tokens before sampling (<=0: no filtering)
    top_p = 0.9  # Nucleus filtering (top-p) before sampling (<=0.0: no filtering)
    max_history = 2

    def __init__(self):
        if self.seed != 0:  # why is this here?
            random.seed(self.seed)
            torch.random.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)

        # build model and tokenizer
        self.model_checkpoint = self.get_pretrained_model()
        self.host_device = "cpu"
        # host_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer_class, model_class = (OpenAIGPTTokenizer, OpenAIGPTLMHeadModel)
        self.tokenizer = self.tokenizer_class.from_pretrained(self.model_checkpoint)
        self.model = model_class.from_pretrained(self.model_checkpoint)
        self.model.to(self.host_device)
        interface.add_special_tokens_(self.model, self.tokenizer)

    # Modifies Hugging Faces download_pretrained_model() that's found in utils.py.
    # Is renamed and uses a permanent directory rather than a temporary one.
    def get_pretrained_model(self):
        if not path.exists(self.model_path):
            """ Download and extract finetuned model from S3 """
            Path(self.model_path).mkdir(parents=True, exist_ok=True)
            resolved_archive_file = cached_path(self.HF_FINETUNED_MODEL)

            # logger.info("extracting archive file {} to temp dir {}".format(resolved_archive_file, tempdir))
            with tarfile.open(resolved_archive_file, 'r:gz') as archive:
                archive.extractall(self.model_path)

        return self.model_path

    def convert_sentence_to_tokens_to_ids(self, a_sentence):
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(a_sentence))

    def mk_personality(self, list_of_sentences):
        return [self.convert_sentence_to_tokens_to_ids(a_sentence) for a_sentence in list_of_sentences]

    # Persona is a list of strings. history an empty list or the history returned from a previous call to respond.
    # Raw_statement is a question or response to a bot in the form as a single string.
    def respond(self, persona, raw_statement, raw_history=[]):
        personality = self.mk_personality(persona)
        response, return_history = None, None  # Initialize return values.

        # Convert string history to IDS (numbers).
        transformed_history = [self.tokenizer.encode(a_sentence) for a_sentence in raw_history]

        if raw_statement:
            transformed_history.append(self.tokenizer.encode(raw_statement))
            with torch.no_grad():
                # At this point it was been encoded into numbers, additional tokens and all.
                out_ids = interface.sample_sequence(personality, transformed_history,
                                                    self.tokenizer, self.model, self.host_device,
                                                    self.temperature, self.top_k, self.top_p,
                                                    self.no_sample, self.max_out_utter, self.min_out_utter)
            transformed_history.append(out_ids)
            transformed_history = transformed_history[-(2 * self.max_history + 1):]
            transformed_history = [self.tokenizer.decode(a_sentence, skip_special_tokens=True)
                                   for a_sentence in transformed_history]
            response = self.tokenizer.decode(out_ids, skip_special_tokens=True)

        return response, transformed_history

    def test_respond(self):
        persona = ["I'm the worlds most decorated olympian.",
                   "The Arizona State Sun Devils are my favorite team. I volunteer there.",
                   "I like to party; no one else likes me partying. But I like to party.",
                   "I also like to swim.",
                   "I have mental health issues, but overcame them with the help of therapy."]
        convo = ["hello! how are you?", "i'm good. what do you like to do for fun?"]

        history = []
        for i in range(len(convo)):
            response, history = self.respond(persona, convo[i], history)
            print(response, history)


