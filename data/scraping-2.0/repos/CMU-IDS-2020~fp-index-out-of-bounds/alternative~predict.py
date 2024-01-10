import torch, os
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
from tqdm import tqdm
import numpy as np
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig, WEIGHTS_NAME, CONFIG_NAME
#from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
from bertviz.bertviz.pytorch_pretrained_bert import BertModel, BertTokenizer

special_tokens = ['<POS>', '<NEG>','<CON_START>','<START>','<END>'] # Set the special tokens
tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt', special_tokens=special_tokens)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt', num_special_tokens=len(special_tokens))
path = os.path.join(os.getcwd(), "./pytorch_model_zero_grad_1.bin") ## Model Path
model_state_dict = torch.load(path, map_location=device)
model.load_state_dict(model_state_dict)
model.to(device)
model.eval()
bert_classifier_dir = "./bert_classifier/" 
model_cls = BertForSequenceClassification.from_pretrained(bert_classifier_dir, num_labels=2)
tokenizer_cls = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
#model_cls.to(device)
#model_cls.eval()
max_seq_len=70
sm = torch.nn.Softmax(dim=-1)
model.config.n_positions

def preditction_with_beam_search(ref_text, beam_width=3, vocab_length=40483):
    """
    This function decodes sentences using Beam Seach. 
    It will output #sentences = beam_width. This function works on a single example.
    
    ref_text : string : Input sentence
    beam_width : int : Width of the output beam
    vocab_length : int : Size of the Vocab after adding the special tokens
    """
    
    done = [False for i in range(beam_width)] # To track which beams are already decoded
    stop_decode = False
    decoded_sentences=[] # List of decoded sentences at any given time
    
    sm = torch.nn.Softmax(dim=-1) # To calculate Softmax over the final layer Logits
    tokens = tokenizer.tokenize(ref_text) # Tokenize the input text
    print(tokens)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokens) # Convert tokens to ids
    index_tokens = [indexed_tokens for i in range(beam_width)] # Replication of Input ids for all the beams

    #index_tokens = [indexed_tokens for i in range(beam_width)]
    torch_tensor = torch.tensor(index_tokens).to(device)
    beam_indexes = [[] for i in range(beam_width)] # indexes of the current decoded beams
    best_scoes = [0 for i in range(beam_width)] # A list of lists to store Probability values of each decoded token of best beams
    count = 0
    while count < model.config.n_positions and not stop_decode:
        if count == 0: # For the first step when only one sentence is availabe
            with torch.no_grad():
                # Calculate output probability distribution over the Vocab,
                preds = sm(model(torch_tensor)) #  shape = [beam_bidth, len(input_sen)+1,Vocab_length]
            top_v, top_i = preds[:,-1,:].topk(beam_width) # Fatch top indexes and it's values
            [beam_indexes[i].append(top_i[0][i].tolist()) for i in range(beam_width)] # Update the Beam indexes
            # Update the best_scores, for first time just add the topk values directly
            for i in range(beam_width):
                best_scoes[i] = top_v[0][i].item()
            count += 1
        else: # After first step
            # Prepare the current_state by concating original input and decoded beam indexes
            current_state = torch.cat((torch_tensor, torch.tensor(beam_indexes).to(device)), dim=1)
            # Prediction on the current state
            with torch.no_grad():
                preds = sm(model(current_state))
            # Multiply new probability predictions with corresponding best scores
            # Total socres = beam_width * Vocab_Size
            flatten_score = (preds[:,-1,:]*torch.tensor(best_scoes).to(device).unsqueeze(1)).view(-1)
            # Fatch the top scores and indexes 
            vals, inx = flatten_score.topk(beam_width)
            # best_score_inx saves the index of best beams after multiplying the probability of new prediction
            best_scoes_inx = (inx // vocab_length).tolist()
            best_scoes = vals.tolist()
            # Unflatten the index 
            correct_inx = (inx % vocab_length).tolist()
            
            # Check if done for all the Beams
            for i in range(beam_width):
                if correct_inx[i] == tokenizer.special_tokens["<END>"]:
                    done[i] = True
            # Update the best score for each the current Beams
            for i in range(beam_width):
                if not done[i]:
                    best_scoes[i] = vals.tolist()[i]
            # Check is All the Beams are Done
            if (sum(done) == beam_width):
                stop_decode = True
            # Prepapre the new beams
            temp_lt=[0 for i in range(beam_width)]
            for i,x in enumerate(best_scoes_inx):
                temp_lt[i] = beam_indexes[x] + [correct_inx[i]]
            # Update the Beam indexes
            beam_indexes = temp_lt
            del temp_lt
            count += 1
    # Decode All the beam indexes to till <END> token only and convert into sentence
    for i in range(beam_width):
        try:
            end_index = beam_indexes[i].index(tokenizer.special_tokens["<END>"])
        except ValueError:
            end_index = len(beam_indexes[i])
            
        decoded_sentences.append(tokenizer.decode(beam_indexes[i][:end_index]))
        
    return decoded_sentences

def get_best_sentence(input_sentences, sentiment=1):
    """
    This function selects the sentence from the Beam of the sentences,
    based on the classification probability score.
    
    input_sentences : list of strings : Sentences generated by the Beam search decoding
    sentiment: int : Expected sentiment (in general class for the classification)
    """
    # BERT pre-processing
    ids = []
    segment_ids = []
    input_masks = []
    pred_lt = []
    for sen in input_sentences:
        text_tokens = tokenizer_cls.tokenize(sen)
        tokens = ["[CLS]"] + text_tokens + ["[SEP]"]
        temp_ids = tokenizer_cls.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(temp_ids)
        segment_id = [0] * len(temp_ids)
        padding = [0] * (max_seq_len - len(temp_ids))

        temp_ids += padding
        input_mask += padding
        segment_id += padding
        
        ids.append(temp_ids)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
    
    ids = torch.tensor(ids).to(device)
    segment_ids = torch.tensor(segment_ids).to(device)
    input_masks = torch.tensor(input_masks).to(device)
    # prediction
    with torch.no_grad():
        preds = sm(model_cls(ids, segment_ids, input_masks))
        
    preds = preds.tolist()
    inx, inx_val = None, 0
    for i in range(len(input_sentences)):
        temp = preds[i][sentiment]
        if temp > inx_val:
            inx = i
            inx_val = temp
    return input_sentences[inx]

op=preditction_with_beam_search("<NEG> <CON_START> there are foods <START>",1)
print(op)
# print(get_best_sentence(op))
