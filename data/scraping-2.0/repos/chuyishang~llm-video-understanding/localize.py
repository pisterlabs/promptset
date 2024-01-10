import os
import string
import json
import torch
import numpy as np
import openai
import random
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
from nemo.collections.nlp.models import PunctuationCapitalizationModel
import argparse
from tqdm import tqdm
#import en_core_web_sm
import spacy
from sentence_transformers import SentenceTransformer, util
import multiprocessing as mp
import _io
from get_times import *

f = open("/home/shang/openai-apikey.txt")
#print(f.readlines()[0])
openai.api_key = f.readlines()[0]

nlp = spacy.load('en_core_web_sm')
#nlp = en_core_web_sm.load()
sent_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-mpnet-base-v2")

def get_next_character(text_list, index1, index2):
    if index1 == len(text_list):
        return None, index1, index2
    if index2 == len(text_list[index1]):
        return get_next_character(text_list, index1+1, 0)
    if text_list[index1][index2].isspace():
        return get_next_character(text_list, index1, index2+1)
    return text_list[index1][index2], index1, index2

def align_after_postprocess(postprocessed, original):
    index_map = {}
    speech_segment_index = 0
    within_segment_index = 0
    p_index = 0
    postprocessed_l = postprocessed # .lower()
    while p_index < len(postprocessed_l):
        if postprocessed_l[p_index].isspace():
            p_index += 1
            continue
        char, speech_segment_index, within_segment_index = get_next_character(original["text"], speech_segment_index, within_segment_index)
        if char is not None:
            _, next_speech_segment_index, next_within_segment_index = get_next_character(original["text"], speech_segment_index, within_segment_index+1)
            if postprocessed_l[p_index].upper().lower() == char.upper().lower() or postprocessed_l[p_index:p_index+2].upper().lower() == char.upper().lower():
                index_map[p_index] = (speech_segment_index, within_segment_index)
                speech_segment_index = next_speech_segment_index
                within_segment_index = next_within_segment_index
        p_index += 1
    return index_map

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def encode_section(sent_model, sents, start, end):
    section = ' '.join(sents[start:end])
    return {(start, end): sent_model.encode([section])[0]}

def remove_punctuation(text):
    new_text = text
    for c in string.punctuation:
        new_text = new_text.replace(c, '')
    return new_text

def align_text(text, original_text, steps, sent_model, num_workers, do_dtw=False, do_drop_dtw=True, dtw_window_size=10000000000, dtw_start_offset=False, id=None):
    #print("===================")
    doc = nlp(text)
    #print("DOC:", doc)
    #print("===================")
    sents = [str(sent) for sent in list(doc.sents)]
    if args.gen_steps:
        steps = args.gen_steps.split("\\n")
    else:
        steps = steps[:len(sents)]
    #("=========================")
    #print("SENTS:", len(sents), sents)
    #print("STEPS:", len(steps), steps)
    #print("=========================")
    step_embs = sent_model.encode(steps)
    text = text.replace('ı', 'i')
    if do_dtw:
        dtw_matrix = np.zeros((len(steps)+1, len(sents)+1, len(sents)+1)) # dtw matrix size [steps+1, sent+1, sent+1]
        #print("==========================")
        #print(dtw_matrix.shape)
        #print("==========================")
        for i in range(len(steps)+1):
            for start in range(len(sents)+1):
                for end in range(len(sents)+1):
                    dtw_matrix[i,start,end] = -np.inf # sets everything to -inf
        dtw_matrix[0,0,0] = 0 # sets start to 0
        pointers = -1*np.ones((len(steps)+1, len(sents)+1, len(sents)+1), dtype=np.int32) # pointers -> pointer matrix (possibly for keeping track of prev values?)
        pointer_scores = -np.inf*np.ones((len(steps)+1, len(sents)+1, len(sents)+1), dtype=np.float32) # pointer_scores -> same size matrix of -infs
        start_sent_index = 0 # sentence start index
        # if there is offset, can ignore for now
        if dtw_start_offset:
            single_sent_emb = np.stack([sent_model.encode([sent])[0,:] for sent in sents])
            start_scores = (step_embs[:1,:]*single_sent_emb).sum(1)
            start_sent_index = min(max(0, start_scores.argmax()-1), len(sents)-len(steps))
            dtw_matrix[0,start_sent_index,start_sent_index] = 0
        # section_emb -> empty dic, we append to it later
        section_emb = {}
        if num_workers == 1:
            batch = [] # batch -> empty arr, we append to it later
            for start in range(start_sent_index, len(sents)): # outer loop: sentences -> for each sentence
                for end in range(start+1, min(start+dtw_window_size+1, len(sents)+1)): # for end in (start to smaller of window size or sentence length)
                    section = ' '.join(sents[start:end]) # section -> joined sentences from start index to end, represents 1 section
                    batch.append((start, end, section)) # append tuple (start index, end index, section) to batch list
                    if len(batch) == 16 or (start == len(sents)-1 and end == len(sents)): # when batch is full:
                        inputs = [item[-1] for item in batch] # inputs -> list of sections (combined sections)
                        outputs = sent_model.encode(inputs) # out -> encoded inputs (sections)
                        for item, output in zip(batch, outputs): # item -> batch, output -> outputs (encoded)
                            section_emb[item[:2]] = output # section_emb[batch[:2]] = --> key is (start, end), val is encoded output, 2 is so that it gets stored in the output section
                        batch = [] # resets batch if previous batch full
            if len(batch) > 0: # if batch nonempty: (this is the TAIL CASE)
                inputs = [item[-1] for item in batch] # inputs = section (-1 is last element)
                outputs = sent_model.encode(inputs) # same process as before
                for item, output in zip(batch, outputs):
                    section_emb[item[:2]] = output
        else:
            with mp.Pool(num_workers) as pool:
                section_emb_list = pool.starmap(encode_section, [(sent_model, sents, start, end) for start in range(0, len(sents)) for end in range(start+1, min(start+dtw_window_size+1, len(sents)+1))])
            for emb_dict in section_emb_list:
                section_emb.update(emb_dict)
        for i in range(1, len(steps)+1): # for step:
            for start in range(start_sent_index, len(sents)): # for start index:
                for end in range(start+1, min(start+dtw_window_size+1, len(sents)+1)): # for end index:
                    #print(f"({i, start, end})")
                    section = ' '.join(sents[start:end]) # section formed by joined sentences
                    sentence_emb = section_emb[(start,end)] # sent_model.encode([section])[0] sentence_emb -> encoded sentence embedding for [start to end]
                    step_emb = step_embs[i-1] # step_emb -> step embedding
                    similarity = (sentence_emb*step_emb).sum().item() # take dot product similarity
                    best_prev_segment = dtw_matrix[i-1,:,start].argmax().item() # [step -1, :, start sentence]
                    #print("BEST PREV SEG:", sha)
                    prev_segment_score = dtw_matrix[i-1,:,start].max().item() # [step -1, :, start sentence]
                    #print("PREV SEGMENT SCORE:", prev_segment_score) -> this is a single number
                    # if prev_segment_score > dtw_matrix[i-1,start,end].item():
                    #     pointers[i,start,end] = best_prev_segment
                    # else:
                    #     pointers[i,start,end] = start
                    pointers[i,start,end] = best_prev_segment
                    pointer_scores[i,start,end] = prev_segment_score
                    last_max = np.max([prev_segment_score]) # , dtw_matrix[i-1,start,end]])
                    dtw_matrix[i,start,end] = similarity+last_max
            # print('good', i, [j for j in range(dtw_matrix.shape[1]) if dtw_matrix[i,j,:].max().item() > -np.inf])
        # sentence - 1
        end = dtw_matrix.shape[1]-1
        # steps - 1
        index = dtw_matrix.shape[0]-1
        start = dtw_matrix[index,:,end].argmax().item()
        #print("=====================")
        #print("MAX:", dtw_matrix[index,:,:end].max().item(), "START", start, "END", end, "INDEX", index)
        #print("=====================")
        segments = {index: (start, end)}
        index -= 1
        while index > 0:
            # print(index+1, start, end)
            new_start = int(pointers[index+1,start,end])
            #print(pointer_scores[index+1,start,end])
            if new_start != start:
                end = start
                start = new_start
            # else:
            #     print('bad', pointers[index+1,start,end], pointer_scores[index+1,start,end])
            segments[index] = (start, end)
            index -= 1
        #print("PRINT!!:", start_sent_index, segments)
    elif do_drop_dtw:
        sent_emb = sent_model.encode(sents)
        #scores = torch.matmul(torch.from_numpy(step_embs), torch.from_numpy(sent_emb).t())
        scores = util.cos_sim(step_embs, sent_emb)
        #print("SENT EMB:", sent_emb.shape, "STEP EMB:", step_embs.shape, "SCORES", scores.shape)
        
        
        def drop_dtw(zx_costs, drop_costs, exclusive=True, contiguous=True, return_labels=False):
            """Drop-DTW algorithm that allows drop only from one (video) side. See Algorithm 1 in the paper.
        
            Parameters
            ----------
            zx_costs: np.ndarray [K, N] 
                pairwise match costs between K steps and N video clips
            drop_costs: np.ndarray [N]
                drop costs for each clip
            exclusive: bool
                If True any clip can be matched with only one step, not many.
            contiguous: bool
                if True, can only match a contiguous sequence of clips to a step
                (i.e. no drops in between the clips)
            return_label: bool
                if True, returns output directly useful for segmentation computation (made for convenience)
            """
            K, N = zx_costs.shape
            
            # initialize solutin matrices
            D = np.zeros([K + 1, N + 1, 2]) # the 2 last dimensions correspond to different states.
                                            # State (dim) 0 - x is matched; State 1 - x is dropped
            D[1:, 0, :] = np.inf  # no drops in z in any state
            D[0, 1:, 0] = np.inf  # no drops in x in state 0, i.e. state where x is matched
            D[0, 1:, 1] = np.cumsum(drop_costs)  # drop costs initizlization in state 1
        
            # initialize path tracking info for each state
            P = np.zeros([K + 1, N + 1, 2, 3], dtype=int) 
            for xi in range(1, N + 1):
                P[0, xi, 1] = 0, xi - 1, 1
            
            # filling in the dynamic tables
            for zi in range(1, K + 1):
                for xi in range(1, N + 1):
                    # define frequently met neighbors here
                    diag_neigh_states = [0, 1] 
                    diag_neigh_coords = [(zi - 1, xi - 1) for _ in diag_neigh_states]
                    diag_neigh_costs = [D[zi - 1, xi - 1, s] for s in diag_neigh_states]
        
                    left_neigh_states = [0, 1]
                    left_neigh_coords = [(zi, xi - 1) for _ in left_neigh_states]
                    left_neigh_costs = [D[zi, xi - 1, s] for s in left_neigh_states]
        
                    left_pos_neigh_states = [0] if contiguous else left_neigh_states
                    left_pos_neigh_coords = [(zi, xi - 1) for _ in left_pos_neigh_states]
                    left_pos_neigh_costs = [D[zi, xi - 1, s] for s in left_pos_neigh_states]
        
                    top_pos_neigh_states = [0]
                    top_pos_neigh_coords = [(zi - 1, xi) for _ in left_pos_neigh_states]
                    top_pos_neigh_costs = [D[zi - 1, xi, s] for s in left_pos_neigh_states]
        
                    z_cost_ind, x_cost_ind = zi - 1, xi - 1  # indexind in costs is shifted by 1
        
                    # state 0: matching x to z
                    if exclusive:
                        neigh_states_pos = diag_neigh_states + left_pos_neigh_states
                        neigh_coords_pos = diag_neigh_coords + left_pos_neigh_coords
                        neigh_costs_pos = diag_neigh_costs + left_pos_neigh_costs
                    else:
                        neigh_states_pos = diag_neigh_states + left_pos_neigh_states + top_pos_neigh_states
                        neigh_coords_pos = diag_neigh_coords + left_pos_neigh_coords + top_pos_neigh_coords
                        neigh_costs_pos = diag_neigh_costs + left_pos_neigh_costs + top_pos_neigh_costs
                    costs_pos = np.array(neigh_costs_pos) + zx_costs[z_cost_ind, x_cost_ind] 
                    opt_ind_pos = np.argmin(costs_pos)
                    P[zi, xi, 0] = *neigh_coords_pos[opt_ind_pos], neigh_states_pos[opt_ind_pos]
                    D[zi, xi, 0] = costs_pos[opt_ind_pos]

                    # state 1: x is dropped
                    costs_neg = np.array(left_neigh_costs) + drop_costs[x_cost_ind] 
                    opt_ind_neg = np.argmin(costs_neg)
                    P[zi, xi, 1] = *left_neigh_coords[opt_ind_neg], left_neigh_states[opt_ind_neg]
                    D[zi, xi, 1] = costs_neg[opt_ind_neg]
        
            cur_state = D[K, N, :].argmin()
            min_cost = D[K, N, cur_state]
                    
            # backtracking the solution
            zi, xi = K, N
            path, labels = [], np.zeros(N)
            x_dropped = [] if cur_state == 1 else [N]
            while not (zi == 0 and xi == 0):
                path.append((zi, xi))
                zi_prev, xi_prev, prev_state = P[zi, xi, cur_state]
                if xi > 0:
                    labels[xi - 1] = zi * (cur_state == 0)  # either zi or 0
                if prev_state == 1:
                    x_dropped.append(xi_prev)
                zi, xi, cur_state = zi_prev, xi_prev, prev_state
            
            return min_cost, path, x_dropped, labels

        drop_cost = np.percentile(-scores.flatten(), 20)
        drop_cost_array = np.ones(len(sents)) * drop_cost
        ddtw_results = drop_dtw(-scores.numpy(), drop_cost_array, contiguous=True)
        
        segs = {}
       
        for s in np.unique(ddtw_results[3]):
            if s==0:
                continue
            indexes = np.where(ddtw_results[3] == s)[0] + 1
            segs[int(s)] = (min(indexes), max(indexes))
        #print("SEGS", segs)
        #print("=======================\n")
        if args.hr_folder:
            human_readable = {}
            for i in segs.keys():
                print(i)
                print(steps)
                step_sentences = []
                for f in range(segs[i][0], segs[i][1] + 1):
                    step_sentences.append(sents[f-1])
                human_readable[i] = step_sentences
                with open(args.hr_folder + id + ".json", "w") as outfile:
                    json.dump(human_readable, outfile)
        segments = dict(reversed(list(segs.items())))
        #print("HUMAN READABLE:", human_readable)
        
    else:
        print("ERROR!")
        return

    if args.allow_drops:
        sent_embs = sent_model.encode(sents)
        sims_arr = [] # [step, sent, sim]
        for index in segments.keys():
            start, end = segments[index]
            #print("INDEX:", index, "STARTEND", start, end)
            #print("SENT EMB SHAPE:", sent_embs.shape)
            step_emb = step_embs[index - 1]
            #print("STEP EMB SHAPE:", step_emb.shape)
            sims = []
            for i in range(start, end):
                #sims.append((index, i, sent_embs[i] @ step_emb))
                sims.append(sent_embs[i] @ step_emb)
            relative_scores = [(b - max(sims)) / max(sims) for b in sims]
            #print(relative_scores)
            heuristic = -0.6
            start_counter, end_counter = 0, 0
            for i in range(1, len(relative_scores)):
                if (relative_scores[i] + relative_scores[i-1])/2 < heuristic:
                    start_counter += 1
                else:
                    break
            for j in range(len(relative_scores)-2, 0, -1):
                if (relative_scores[j] + relative_scores[j+1])/2 < heuristic:
                    end_counter += 1
                else:
                    break
            #print("INDEX", index, "START,END COUNTER:", start_counter, end_counter)
            segments[index] = (start + start_counter, end - end_counter)
            print(f"PROCESSED SEGMENT {index}")
            #sims_arr.append(sims)
        #print(sims_arr)

            

    postprocess_alignment = align_after_postprocess(text, original_text)
    # print(segments)
    # print(postprocess_alignment)
    aligned_segments = {}
    sents = list(doc.sents)
    #print("====================")
    #print("SEGMENTS:", segments)
    #print("POSTPROC ALIGN:", postprocess_alignment)
    #print("====================")
    # print(text)
    # print(original_text)
    # print(' '.join(original_text['text']))
    # print(max(list(postprocess_alignment.keys())), [sents[segments[index][0]].start_char for index in segments], [text[sents[segments[index][0]].start_char:sents[segments[index][1]-1].end_char] for index in segments])
    for index in segments:
        #print("DEBUG:", segments[index][0], len(sents), sents)
        
        # TEMP FIX
        TEMP_FIX_INDEX = segments[index][0]
        if segments[index][0] == len(sents):
            TEMP_FIX_INDEX -= 1

        while str(sents[TEMP_FIX_INDEX]).isspace():
            #print(f"===========\n{index}\n==============\n")
            segments[index] = (segments[index][0]-1, segments[index][1])
        # TEMP FIX:
        start = sents[TEMP_FIX_INDEX].start_char
        #print("================")
        #print("START:", start)
        #print("================")
        while start not in postprocess_alignment and start < len(text):
            start += 1
        if start not in postprocess_alignment:
            print('A', sents[segments[index][0]])
            print('B', text[sents[segments[index][0]].start_char:], sents[segments[index][0]].start_char)
            print('C', text)
            print('D', ' '.join(original_text['text']))
            print(sents[segments[index][0]].start_char, sorted(list(postprocess_alignment.keys()))[-50:])
        assert start in postprocess_alignment
        end = sents[segments[index][1]-1].end_char-1
        while end not in postprocess_alignment and end >= 0:
            end -= 1
        assert end in postprocess_alignment
        #print("=============")
        #print("POSTPROC START", postprocess_alignment[start], "POSTPROC END",  postprocess_alignment[end])
        #print("=============")
        aligned_segments[index] = postprocess_alignment[start]+postprocess_alignment[end]
        #print("==================")
        #print('ALIGNED:', ' '.join(original_text['text'][aligned_segments[index][0]:aligned_segments[index][2]+1]), sents[segments[index][0]:segments[index][1]])
        #print("==================")
    return aligned_segments

def remove_repeat_ngrams(text_list, min_n=3, max_n=8, return_segment_ids=False):
    assert isinstance(text_list, list)
    tokens = []
    segment_ids = []
    for segment_id, segment in enumerate(text_list):
        segment_tokens = segment.split()
        for token in segment_tokens:
            if len(token) > 0:
                tokens.append(token)
                segment_ids.append(segment_id)
    inside_segment = False
    num_streak_tokens = 0
    new_tokens = []
    new_segment_ids = []
    indices_added = set()
    for i in range(len(tokens)):
        redundant = False
        for j in range(max_n, min_n-1, -1):
            if i+1 >= j*2 and tokens[i+1-j:i+1] == tokens[i+1-j*2:i+1-j]:
                # print('here', tokens[i+1-j*2:i+1])
                inside_segment = True
                num_streak_tokens = min_n
                for k in range(1, j):
                    if i-k in indices_added:
                        new_tokens.pop()
                        new_segment_ids.pop()
                        indices_added.remove(i-k)
                redundant = True
                break
        if not redundant:
            new_tokens.append(tokens[i])
            indices_added.add(i)
            new_segment_ids.append(segment_ids[i])
    if return_segment_ids:
        return ' '.join(new_tokens), new_segment_ids
    return ' '.join(new_tokens)

def process_video(video_id, args, input_steps, transcripts, tokenizer, punct_cap_model, output_queue):
    prompt = "Write the steps of the task that the person is demonstrating, based on the noisy transcript.\nTranscript: |||1\nSteps:\n1."
    #print('here3')
    if transcripts is not None:
        try:
            original = transcripts[video_id]
        except:
            print(video_id)
            return
    else:
        f = open(os.path.join(args.transcripts_path, video_id+".csv"))
        lines = f.readlines()
        original = {"text": [], "start": [], "end": []}
        for line in lines[1:]:
            parts = line.split(',')
            original["start"].append(float(parts[0]))
            original["end"].append(float(parts[1]))
            original["text"].append(parts[-1].strip())
    transcript = " ".join(original["text"])
    deduplicated_text, new_segment_ids = remove_repeat_ngrams(original["text"], min_n=3, max_n=9, return_segment_ids=True)
    deduplicated_tokens = deduplicated_text.split()
    original["text"] = [[] for _ in range(len(original["text"]))]
    for token, new_id in zip(deduplicated_tokens, new_segment_ids):
        original["text"][new_id].append(token)
    original["text"] = [" ".join(lst) for lst in original["text"]]
    transcript = " ".join(original["text"])
    if not args.no_formatting:
        if args.formatted_transcripts_path is not None:
            fname = os.path.join(args.formatted_transcripts_path, video_id+".txt")
        if args.formatted_transcripts_path is not None and os.path.exists(fname):
            f = open(fname)
            transcript = f.readlines()[0]
        else:
            transcript = punct_cap_model.add_punctuation_capitalization([transcript])[0]
    tokens = tokenizer(transcript)
    #print(video_id, len(transcript), len(tokens["input_ids"]))
    while len(tokens["input_ids"]) > 1600:
        transcript = transcript[:-100]
        tokens = tokenizer(transcript)
    
    if args.gen_steps is not None:
        steps = args.gen_steps.split("\\n")
    elif args.input_steps_path is not None:
        if video_id not in input_steps:
            return
        steps = input_steps[video_id]["steps"]
    else:
        if video_id in finished:
            return
        input_text = prompt.replace("|||1", transcript)
        steps = []
        num_attempts = 0
        while len(steps) == 0:
            response = openai.Completion.create(
                            engine="text-babbage-001",
                            prompt=input_text,
                            temperature=0.7,
                            max_tokens=256,
                            top_p=1,
                            frequency_penalty=0,
                            presence_penalty=0
                        )
            output = response["choices"][0]["text"].strip()
            num_attempts += 1
            steps = output.split("\\n")
            if all(["." in step for step in steps[1:]]):
                steps = steps[:1]+[step[step.index(".")+1:].strip() for step in steps[1:]]
            elif num_attempts < args.max_attempts:
                steps = []
    output_dict = {"video_id": video_id, "steps": steps, "transcript": transcript}
    if not args.no_align:
        segments = align_text(transcript, original, steps, sent_model, args.num_workers, args.do_dtw, args.do_drop_dtw, args.dtw_window_size, id=video_id)
        #print(segments)
        output_dict["segments"] = segments
    if isinstance(output_queue, _io.TextIOWrapper):
        output_queue.write(json.dumps(output_dict)+'\n')
    else:
        output_queue.put(json.dumps(output_dict)+'\n')

def output_listener(output_queue, output_filename):
    mode = 'a+' if os.path.exists(output_filename) else 'w'
    with open(output_filename, 'a+') as fout:
        while True:
            output = output_queue.get()
            if output == 'kill':
                break
            fout.write(output)
            fout.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_list_path")
    parser.add_argument("--transcripts_path")
    parser.add_argument("--formatted_transcripts_path")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--max_attempts", type=int, default=1)
    parser.add_argument("--no_formatting", action="store_true")
    parser.add_argument("--output_path")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no_align", action="store_true")
    parser.add_argument("--input_steps_path", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--do_dtw", action="store_true")
    parser.add_argument("--dtw_window_size", type=int, default=1000000)
    parser.add_argument("--allow_drops", action="store_true")
    parser.add_argument("--do_drop_dtw", action="store_true")
    parser.add_argument("--drop_cost_pct", default=25)
    parser.add_argument("--hr_folder")
    parser.add_argument("--gen_steps", type=str, default=None)
    args = parser.parse_args()

    if not args.no_align:
        if args.cpu:
            sent_model = SentenceTransformer('sentence-transformers/paraphrase-mpnet-base-v2').cpu()
        else:
            sent_model = SentenceTransformer('sentence-transformers/paraphrase-mpnet-base-v2').cuda()
    # sent_model = AutoModel.from_pretrained('sentence-transformers/paraphrase-mpnet-base-v2').cuda()
    if not args.no_formatting:
        punct_cap_model = PunctuationCapitalizationModel.from_pretrained("punctuation_en_bert")
        #punct_cap_model = None
        if args.cpu:
            punct_cap_model = punct_cap_model.cpu()
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    f = open(args.video_list_path)
    lines = f.readlines()[0]
    #print("LINES:", lines)
    video_ids = lines.split(",")
    #video_ids = [line.strip().split()[0].split('.')[0] for line in lines]
    transcripts = None
    if args.transcripts_path[-5:] == ".json":
        f = open(args.transcripts_path)
        transcripts = json.load(f)
    if args.end_index is not None:
        video_ids = video_ids[:args.end_index]
    video_ids = video_ids[args.start_index:]
    finished = set()
    if os.path.exists(args.output_path):
        fout = open(args.output_path)
        written_lines = fout.readlines()
        fout.close()
        for line in written_lines:
            try:
                datum = json.loads(line)
                finished.add(datum['video_id'])
            except:
                pass
        fout = open(args.output_path, 'a')
    else:
        fout = open(args.output_path, 'w')
    input_steps = None
    if args.input_steps_path is not None:
        f = open(args.input_steps_path)
        lines = f.readlines()
        input_steps = [json.loads(line) for line in lines]
        input_steps = {datum["video_id"]: datum for datum in input_steps}
    """manager = mp.Manager()
    q = manager.Queue()
    pool = mp.Pool(args.num_workers+2)
    watcher = pool.apply_async(output_listener, (q, args.output_path))
    print('here1', pool._processes)
    jobs = []"""
    for video_id in tqdm(video_ids):
        if video_id in finished:
            continue
        # job = pool.apply_async(process_video, (video_id, args, input_steps, transcripts, tokenizer, punct_cap_model, q))
        process_video(video_id, args, input_steps, transcripts, tokenizer, punct_cap_model, fout)
        # print('here', len(jobs))
        # jobs.append(job)
    """for job in jobs:
        job.get()
    q.put('kill')
    pool.close()
    pool.join()"""
    fout.close()
