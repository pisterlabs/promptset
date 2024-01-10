import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import logging
from torchmetrics.functional import bleu_score  # for model evaluation
from transformers import MarianMTModel, MarianTokenizer
import openai

class LanguageUnderstandingModule:
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = api_key
        self.language_models = {}  # define this attribute


    def add_language(self, lang, model_name):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = MarianMTModel.from_pretrained(model_name).to(device)
        self.language_models[lang] = (model, MarianTokenizer.from_pretrained(model_name))


    def translate_to_plot(self, texts, lang):
        model = self.language_models[lang]
        tokenizer = model.config.tokenizer_class.from_pretrained(model_name)
        input_tokens = tokenizer.prepare_seq2seq_batch(src_texts=texts, return_tensors="pt")
        input_tokens = {k: v.to(model.device) for k, v in input_tokens.items()}
        output_tokens = model.generate(**input_tokens)
        return [tokenizer.decode(tok, skip_special_tokens=True) for tok in output_tokens]

    def load_model(self, model_name, language):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = MarianMTModel.from_pretrained(model_name).to(device)
        self.language_models[language] = model


    def translate_to_plot(self, texts, lang):
        openai.api_key = self.api_key
        model_engine = "text-davinci-002"  # Use the desired GPT-3 model engine

        prompt = "Translate the following text(s) to PLoT expressions:\n"
        for text in texts:
            prompt += f"Text: {text}\n"
        prompt += "End of texts."

        response = openai.Completion.create(engine=model_engine, prompt=prompt, max_tokens=150, n=1, stop=None, echo=False, temperature=0.7)
        generated_text = response.choices[0].text
        plot_expressions = generated_text.strip().split("\n")

        return plot_expressions  # remember to return the result

    def compute_confidence_scores(self, output_tokens, lang):
        model = self.model_mapping[lang]
        tokenizer = self.tokenizer_mapping[lang]
        input_tokens = tokenizer(output_tokens.tolist(), return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**input_tokens, labels=input_tokens["input_ids"])
        confidence_scores = -outputs.loss * input_tokens["input_ids"].shape[1]
        return confidence_scores.tolist()

    def add_language(self, lang, model_name):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = MarianMTModel.from_pretrained(model_name).to(device)
        self.language_models[lang] = model

    def request_clarification(self, text):
        print(f"The system is unsure about the following text: '{text}'")
        user_clarification = input("Could you please provide more context or rephrase? ")
        return user_clarification

    def fallback(self, text):
        print(f"Translation of the following text failed: '{text}'")
        return text

    
    def train_custom_model(self, training_data, model_name, language):
        model = self.model_mapping[language]
        tokenizer = self.tokenizer_mapping[language]
        inputs, targets = zip(*training_data)
        input_tokens = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
        target_tokens = tokenizer(targets, return_tensors="pt", padding=True, truncation=True)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        for epoch in range(NUM_EPOCHS):
            outputs = model(**input_tokens)
            loss = loss_fn(outputs.logits.view(-1, outputs.logits.size(-1)), target_tokens["input_ids"].view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f"Epoch {epoch}, Loss: {loss.item()}")

        # Save the model for later use (replace 'path/to/save' with actual path)
        torch.save(model.state_dict(), 'path/to/save')

    def handle_error(self, error):
        if isinstance(error, ValueError):
            print("A ValueError occurred:", error)
        elif isinstance(error, RuntimeError):
            print("A RuntimeError occurred:", error)
        else:
            print("An unexpected error occurred:", error)

    def log(self, message, level=logging.INFO):
        if level == logging.INFO:
            self.logger.info(message)
        elif level == logging.WARNING:
            self.logger.warning(message)
        elif level == logging.ERROR:
            self.logger.error(message)
        else:
            self.logger.debug(message)

    def evaluate(self, test_data, lang):
        # Test data is a list of tuples (input_text, target_text)
        model = self.model_mapping[lang]
        tokenizer = self.tokenizer_mapping[lang]
        inputs, targets = zip(*test_data)
        input_tokens = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
        target_tokens = tokenizer(targets, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            output_tokens = model.generate(**input_tokens)
        predictions = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
        # Compute BLEU score
        bleu = bleu_score(predictions, targets)
        print(f"BLEU score: {bleu}")

# Example usage:
model_name = "Helsinki-NLP/opus-mt-en-ROMANCE"
lum = LanguageUnderstandingModule('your-api-key')
lum.add_language("en-fr", model_name)  # English to French translation
texts = ["Hello, how are you?", "What's your name?"]
