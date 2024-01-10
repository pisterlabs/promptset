from transformers import GPT2TokenizerFast, GPT2Tokenizer, GPT2LMHeadModel, GPT2Model, GPT2Config
from transformers import XLMTokenizer, XLMModel
from transformers import BertForMaskedLM, BertTokenizer, BertTokenizerFast, BertConfig, BertModel
from transformers import DistilBertTokenizer, DistilBertTokenizerFast, DistilBertForMaskedLM, DistilBertModel 
from transformers import OpenAIGPTTokenizerFast, OpenAIGPTTokenizer, OpenAIGPTLMHeadModel, OpenAIGPTModel

def get_model_tokenizer(model_name, lm=False):
    # build model
    if not lm:
        if model_name == 'gpt2':
            model = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        if model_name == 'bert':
            model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True, output_hidden_states=True)
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        if model_name == 'dist':
            model = DistilBertModel.from_pretrained('distilbert-base-uncased', output_attentions=True, output_hidden_states=True)
            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        if model_name == 'gpt':
            model = OpenAIGPTModel.from_pretrained('openai-gpt', output_attentions=True, output_hidden_states=True)
            tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
        if model_name == 'xlm':
            model = XLMModel.from_pretrained('xlm-clm-enfr-1024', output_attentions=True, output_hidden_states=True)
            tokenizer = XLMTokenizer.from_pretrained('xlm-clm-enfr-1024')
        
        # large models
        if model_name == 'gpt2-medium':
            model = GPT2Model.from_pretrained('gpt2-medium', output_attentions=True, output_hidden_states=True)
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
        if model_name == 'gpt2-large':
            model = GPT2Model.from_pretrained('gpt2-large', output_attentions=True, output_hidden_states=True)
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
        if model_name == 'gpt2-xl':
            model = GPT2Model.from_pretrained('gpt2-xl', output_attentions=True, output_hidden_states=True)
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
        if model_name == 'bert-large':
            model = BertModel.from_pretrained('bert-large-uncased', output_attentions=True, output_hidden_states=True)
            tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

    if lm:
        if model_name == 'gpt2':
            model = GPT2LMHeadModel.from_pretrained('gpt2')
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        if model_name == 'bert':
            model = BertForMaskedLM.from_pretrained('bert-base-uncased')
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        if model_name == 'dist':
            model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        if model_name == 'gpt':
            model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
            tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
        if model_name == 'xlm':
            tokenizer = XLMTokenizer.from_pretrained('xlm-clm-enfr-1024')
            model = XLMModel.from_pretrained('xlm-clm-enfr-1024')

    # special handle gpt models
    if model_name == 'gpt2' or model_name == 'gpt':
        # gpt2 does not have pad token, need to fix in this way
        # Add the <pad> token to the vocabulary
        SPECIAL_TOKENS = {'pad_token': "<pad>"}
        tokenizer.add_special_tokens(SPECIAL_TOKENS)
        model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def get_special(tokenizer):
    if isinstance(tokenizer, XLMTokenizer):
        special_dict = tokenizer.special_tokens_map # .keys()[:-1] # the last one is additional_special_tokens, should be removed in TODO
    else:
        special_dict = tokenizer.special_tokens_map # special tokens and their id
    return special_dict
