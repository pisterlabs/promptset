from pytorch_pretrained_bert.tokenization import BertTokenizer
from lib.bert_cls.nli_task import main as nli_engine
import os
from pytorch_pretrained_bert import CONFIG_NAME, BertModel
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig, WEIGHTS_NAME
import torch
import logging
logger = logging.getLogger(__file__)
from rl_utils import create_critic, f1_rewarder, bleu_rewarder, plot_reward, LinearRegressionModel, process_document, \
    init_stop_words, read_model, tokens_to_vector, bert_vector
from sklearn.metrics.pairwise import cosine_similarity
from rep_utils import get_ngrams, intrep_frac, flatten, extrep_frac
from pytorch_pretrained_bert.modeling_openai import OpenAIGPTConfig
from pytorch_pretrained_bert import OpenAIAdam, OpenAIGPTTokenizer, GPT2Tokenizer, WEIGHTS_NAME, CONFIG_NAME, BertModel, OpenAIGPTLMHeadModel
from utils import  QN_WORDS

class Rewarder():

    def __init__(self,args,tokenizer):

        self.args = args

        self.nli_tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case, cache_dir='.pytorch_pretrained_bert')
        self.output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        self.output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        self.nli_config = BertConfig(self.output_config_file)
        self.nli_model = BertForSequenceClassification(self.nli_config, num_labels=3)
        self.nli_model.load_state_dict(torch.load(self.output_model_file, map_location=torch.device('cpu')))
        self.nli_model.to(args.device)
        self.nli_model.eval()

        if args.nli_uu_reward or args.nli_allres_reward:
            uu_output_config_file = os.path.join(args.uu_output_dir, CONFIG_NAME)
            uu_output_model_file = os.path.join(args.uu_output_dir, WEIGHTS_NAME)
            self.uu_nli_config = BertConfig(uu_output_config_file)
            self.uu_nli_model = BertForSequenceClassification(self.uu_nli_config, num_labels=3)
            self.uu_nli_model.load_state_dict(torch.load(uu_output_model_file, map_location=torch.device('cpu')))
            self.uu_nli_model.to(args.device)
            self.uu_nli_model.eval()


        bert_emb_modelpath = "bert-base-uncased"
        self.bert_emb_tokenizer = BertTokenizer.from_pretrained(bert_emb_modelpath, cache_dir='.pytorch_pretrained_bert')
        self.bert_emb_model = BertModel.from_pretrained(bert_emb_modelpath, cache_dir='.pytorch_pretrained_bert').to(
            args.device)
        self.bert_emb_model.eval()

        self.tokenizer = tokenizer

        if args.lm_reward:
            lm_model_path = 'openai-gpt'
            lm_output_dir = 'language-quality-subreward/gpt_output'
            lm_special_tokens = ['_start_', '_delimiter_', '_classify_']
            # Load pre-trained model (weights)
            with torch.no_grad():
                lm_output_config_file = os.path.join(lm_output_dir, CONFIG_NAME)
                lm_config = OpenAIGPTConfig(lm_output_config_file)

                lm_output_model_file = os.path.join(lm_output_dir, WEIGHTS_NAME)
                #lm_model_state_dict = torch.load(lm_output_model_file)
                lm_model_state_dict = torch.load(lm_output_model_file, map_location='cpu')
                self.lm_model = OpenAIGPTLMHeadModel(lm_config)
                self.lm_model.load_state_dict(lm_model_state_dict)

                # Load pre-trained model tokenizer (vocabulary)
                self.lm_tokenizer = OpenAIGPTTokenizer.from_pretrained(lm_model_path, special_tokens=lm_special_tokens,
                                                                  cache_dir='.pytorch_pretrained_bert')

            self.special_tokens_ids = list(self.lm_tokenizer.convert_tokens_to_ids(token) for token in lm_special_tokens)
            self.lm_model.to(args.device)
            self.lm_model.eval()

    def persona_rewarder(self, response, rl_train_personas_org):
        # cancat all the personas
        '''
        personas_org_chain = [''.join(rl_train_personas_org)]
        reward = nli_engine(response, personas_org_chain, nli_tokenizer, nli_model)[0]
        '''
        scores = nli_engine(response, rl_train_personas_org, self.nli_tokenizer, self.nli_model)
        current_persona_reward_0 = ((sum(scores) / len(rl_train_personas_org)) + 2) / 3
        current_persona_reward = current_persona_reward_0 * self.args.nli_weight
        logger.info('persona_reward before/after weighting = %f/%f'%(current_persona_reward_0, current_persona_reward))
        return current_persona_reward

    def nli_allres_rewarder(self, response, history):
        # history_chain = list(chain(*history))
        # history_text = tokenizer.decode(history_chain, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        pre_responses = []
        for i in range(-len(history), 0):
            if i % 2 == 0:
                current_text = self.tokenizer.decode(history[i], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                pre_responses.append(current_text)
        response_scores = nli_engine(response, pre_responses, self.nli_tokenizer, self.nli_model)
        if response_scores == []:
            current_response_reward = 0.5  # TODO: test if single allres will work
        else:
            current_response_reward = sum(response_scores) / len(response_scores)
        current_response_reward_0 = (current_response_reward + 2) / 3
        current_response_reward = current_response_reward * self.args.nli_allres_weight
        logger.info('allres_reward before/after weighting = %f/%f'%(current_response_reward_0, current_response_reward))
        return current_response_reward

    def cos_sim_bert_rewarder(self, response, history):
        pre_utt = history[-1]
        pre_utt_text = self.tokenizer.decode(pre_utt, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        pre_utt_vec = bert_vector(pre_utt_text, self.bert_emb_tokenizer, self.bert_emb_model, self.args)
        response_vec = bert_vector(response, self.bert_emb_tokenizer, self.bert_emb_model, self.args)
        cos_sim_bert_score = cosine_similarity(pre_utt_vec.reshape(1, -1), response_vec.reshape(1, -1))[0][0]
        current_cos_sim_bert_reward = cos_sim_bert_score * self.args.cos_sim_bert_weight
        logger.info('cos_sim_bert before/after weighting = %f/%f'%(cos_sim_bert_score,current_cos_sim_bert_reward))
        return current_cos_sim_bert_reward

    def intern_rep_rewarder(self, response):
        # response = 'i\'m 16 years years years years years old bye bye.'
        # intrep_word
        response_tok = response.split()
        intrep_1gram = intrep_frac(response_tok)
        # intrep_2gram
        response_tok_2gram = get_ngrams(response, 2)
        intrep_2gram = intrep_frac(response_tok_2gram)
        # intrep_3gram
        response_tok_3gram = get_ngrams(response, 3)
        intrep_3gram = intrep_frac(response_tok_3gram)
        current_intern_rep_reward = (1 - intrep_1gram) * self.args.intern_rep_weight  # TODO: How to design this reward?
        logger.info('intern_rep before/after weighting = %f/%f'%((1 - intrep_1gram), current_intern_rep_reward))
        return current_intern_rep_reward

    def extern_rep_rewarder(self, response, history):
        pre_responses = []
        for i in range(-len(history), 0):
            if i % 2 == 0:
                current_text = self.tokenizer.decode(history[i], skip_special_tokens=True,
                                                clean_up_tokenization_spaces=False)
                pre_responses.append(current_text)

        # extrep_word
        response_tok = response.split()
        prev_tok = [s.split() for s in pre_responses]  # list of list of ints
        prev_tok = list(set(flatten(prev_tok)))  # list of ints, no duplicates
        extrep_1gram = extrep_frac(response_tok, prev_tok)
        # extrep_2gram
        response_tok_2gram = get_ngrams(response, 2)
        prev_2grams = [get_ngrams(prev, 2) for prev in pre_responses]  # list of list of strings
        prev_2grams = list(set(flatten(prev_2grams)))  # list of strings, no duplicates
        extrep_2gram = extrep_frac(response_tok_2gram, prev_2grams)
        # extrep_3gram
        response_tok_3gram = get_ngrams(response, 3)
        prev_3grams = [get_ngrams(prev, 3) for prev in pre_responses]  # list of list of strings
        prev_3grams = list(set(flatten(prev_3grams)))  # list of strings, no duplicates
        extrep_3gram = extrep_frac(response_tok_3gram, prev_3grams)

        current_extern_rep_reward = 0  # TODO: How to design this reward?
        logger.info('extern_rep before/after weighting = %f/%f'%(current_extern_rep_reward,current_extern_rep_reward))
        return current_extern_rep_reward

    def lm_rewarder(self, response):
        lm_tokenize_input = self.lm_tokenizer.tokenize(response)
        # lm_tensor_input = torch.tensor([lm_tokenizer.convert_tokens_to_ids(lm_tokenize_input)]).to(args.device)
        lm_tensor_input = torch.tensor([[self.special_tokens_ids[0]] + self.lm_tokenizer.convert_tokens_to_ids(lm_tokenize_input) + [self.special_tokens_ids[-1]]]).to(self.args.device)
        lm_loss = self.lm_model(lm_tensor_input, lm_labels=lm_tensor_input)
        # lm_ppl = math.exp(lm_loss.item())
        nll = - lm_loss.item()
        if nll < -4:
            nll = -4
        current_lm_score = (nll + 4) / 4
        current_lm_reward = current_lm_score * self.args.lm_weight  # TODO: 1/lm_ppl?
        logger.info('lm_reward before/after weighting = %f/%f'%(current_lm_score, current_lm_reward))
        return current_lm_reward

    def qback_rewarder(self, response):
        response_tok = response.split()
        num_in_list = len([w for w in response_tok if w in QN_WORDS])
        current_qback_reward = (num_in_list / len(response_tok)) * self.args.qback_weight
        logger.info('qback_reward before/after weighting = %f/%f'%((num_in_list / len(response_tok)), current_qback_reward))
        return current_qback_reward

    def get_reward(self, response, rl_train_personas_org, history):

        R = {'reward': 0,
             'persona_reward': 0,
             'response_reward': 0,
             'uu_reward' : 0,
             'cos_sim_bert_reward':0,
             'intern_rep_reward': 0,
             'extern_rep_reward': 0,
             'lm_reward': 0,
             'qback_reward': 0,
             'f1_reward': 0,
             'bleu_reward': 0}

        if self.args.nli_reward:
            R['persona_reward'] = self.persona_rewarder(response, rl_train_personas_org)

        if self.args.nli_allres_reward:
            R['response_reward'] = self.nli_allres_rewarder(response, history)

        if self.args.nli_uu_reward:
            R['uu_reward'] = self.nli_uu_rewarder(response, history)

        if self.args.cos_sim_bert_reward:
            R['cos_sim_bert_reward'] = self.cos_sim_bert_rewarder(response, history)

        if self.args.intern_rep_reward:
            R['intern_rep_reward'] = self.intern_rep_rewarder(response)

        if self.args.extern_rep_reward:
            R['extern_rep_reward'] = self.extern_rep_rewarder(response, history)

        if self.args.lm_reward:
            R['lm_reward'] = self.lm_rewarder(response)

        if self.args.qback_reward:
            R['qback_reward'] = self.qback_rewarder(response)

        R['reward'] = R['persona_reward'] + \
             R['response_reward'] + \
             R['uu_reward'] + \
             R['cos_sim_bert_reward']+ \
             R['intern_rep_reward'] + \
             R['extern_rep_reward'] + \
             R['lm_reward'] + \
             R['qback_reward'] + \
             R['f1_reward'] + \
             R['bleu_reward']

        return R

