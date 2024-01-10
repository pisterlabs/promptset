#/usr/bin/env python3

import numpy as np
import pandas as pd
import sys, os
import json
import argparse
from tqdm import tqdm

from nltk.tokenize import sent_tokenize, word_tokenize
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

import time

try:
  from IPython import embed
except:
  pass

import openai
openai.api_key = open(os.path.join(os.path.expanduser("~"),".openai_key")).read().strip()

import requests
tnlg_url = "https://turingnlg-uw.turingase.p.azurewebsites.net/freeform/inference?key=56016fb4788644839e2af13144e271fb-maarten"

from extractLinearity import betterSentTokenization, \
  DEFAULT_STORY_ID_COL, DEFAULT_STORY_ID_VALUE, FEAT_COLS, SENT_FEAT_COLS, \
  loadInput, saveOutput, splitIntoSentences, meltFeaturesPerSentence

MAX_SEQ_LEN = 1024

log = logging.getLogger(__file__)
# log.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
#                 datefmt='%Y-%m-%d %H:%M:%S')
# embed();exit()

def getTNLGprobs(toks,text_ix,tok,max_seq_len=MAX_SEQ_LEN,attempt=0):
  if attempt > 10:
    return None
  params = dict(prompt=toks,return_token_logprobs=True,response_length=1,timeout=60)
  response = requests.request("POST", tnlg_url, data=params)
  try:
    r = response.json()["result"]
  except:
    print(response.content)
    if "gateway time-out" in response.content.decode().lower() or response.status_code==504:
      print("Have to restart the machine")
      r = input("Did you restart the machine?")
    return getTNLGprobs(toks,text_ix,tok,max_seq_len=max_seq_len,attempt=attempt+1)
      
  tok_lps = r["token_logprobs"][:-1]
  tok_txt = r["prompt_tokens"]

  # discarding the tokens in the history / context
  # in_sent_txt = tok_txt[text_ix:]
  in_sent_lps = tok_lps[text_ix:]

  return np.array(in_sent_lps)



def getGPT3probs(toks,text_ix,variant="ada",max_seq_len=MAX_SEQ_LEN,attempt=0):
  if attempt > 10:
    return None
  
  if variant == "ada":
    time.sleep(0.04)
  try:
    r = openai.Completion.create(engine=variant,prompt=toks,max_tokens=1,echo=True,logprobs=1)
  except openai.error.InvalidRequestError as e:
    print(e)
    print(x)
    return None
  except (openai.error.RateLimitError, openai.error.APIError,
          openai.error.APIConnectionError, openai.error.OpenAIError,
          openai.error.TryAgain, openai.error.OpenAIErrorWithParamCode) as e:
    print("Got Rate limit error, sleeping for 1 minute",e)
    time.sleep(60)
    return getGPT3probs(toks,text_ix,variant=variant,max_seq_len=max_seq_len,attempt=attempt+1)
    
  a = r["choices"][0]
  tok_lps = a["logprobs"]["token_logprobs"]
  tok_txt = a["logprobs"]["tokens"]  
  text_offset = np.array(a["logprobs"]["text_offset"])
  ix = ((text_offset >= text_ix).cumsum() == 1).argmax()
  # embed();exit()
  # discarding the tokens in the history / context
  in_sent_txt = tok_txt[ix:-1]
  in_sent_lps = tok_lps[ix:-1]

  return np.array(in_sent_lps)


def computePplxInContext(df,model,story_col="story",context_col="summary",
                         sent_col="sents",hist_size=-1,max_seq_len=MAX_SEQ_LEN):
  """
  Computes the xent of tokens in a sentence conditioned on history and context:
    p(s_i|context, s_i-k,...,s_i-1)
  context: text to be prepended to every sentence.
  hist_size: how far back does history go? If -1, it goes back all the way.
  """

  if context_col == "firstSent":
    log.info("Using first sentence as context")
    sents_backup = df[sent_col].copy()
    df[context_col] = df[sent_col].apply(lambda x: x[0])
    df[sent_col] = df[sent_col].apply(lambda x: x[1:])
    
  # Todo: make sure index is unique
  assert len({i for i in df.index}) == len(df), "df index isn't unique"
  dataD = {}
  for ix, ss, cont in df[[sent_col,context_col]].itertuples():
    for i,s in enumerate(ss):
      hist_ix = 0 if hist_size==-1 else max(0,i-hist_size)
      dataD[(ix,i)] = {
        "cont": cont,
        "hist": ss[hist_ix:i],
        "text": s
      }
  data = pd.DataFrame(dataD).T
  data["contHist"] = data["cont"] + "\n\n" + data["hist"].apply(
    lambda x: " ".join(x) + (" " if len(x) > 0 else ""))
  data["contHist"] = data["contHist"].str.lstrip()

  data["toks"] = data["contHist"] + data["text"]
  data["text_ix"] = data["contHist"].apply(len)

  # data["text_xent"] = np.nan
  # data["perTextTokenLogP"] = data["text_xent"].apply(lambda x: [])


  if model == "tnlg":
    gpt2_tok = AutoTokenizer.from_pretrained("gpt2")
    tqdm.pandas(ascii=True,desc="Converting to tokens")
    data["contHistToks"] = data["contHist"].progress_apply(gpt2_tok.encode)
    data["contHistTokLenPlusOne"] = data["contHistToks"].apply(len)+1
    tqdm.pandas(ascii=True,desc="Getting probabilities")
    data["perTextTokenLogP"] = data[["toks","contHistTokLenPlusOne"]].progress_apply(
      lambda x: getTNLGprobs(*x,tok=gpt2_tok,max_seq_len=max_seq_len),axis=1)
  else:
    tqdm.pandas(ascii=True,desc="Getting probabilities")
    data["perTextTokenLogP"] = data[["toks","text_ix"]].progress_apply(
      lambda x: getGPT3probs(*x,variant=model,max_seq_len=max_seq_len),axis=1)
  
  data["text_xent"] = -data["perTextTokenLogP"].apply(np.mean)
  
  data["text_prob"] = data["perTextTokenLogP"].apply(
    lambda x: np.exp(x)).apply(np.mean)
  # data[["perTextTokenLogP","text_xent"]] = data[["toks","text_ix"]].progress_apply(
  #   lambda x: _computePplxInContext(*x,model,tok),axis=1)

  # re-format to make short again
  def reMergeXents(c):
    return pd.Series({
      "text_xents": c["text_xent"].tolist(),
      "text_probs": c["text_prob"].tolist(),
      "perTextTokenLogPs": c["perTextTokenLogP"].tolist(),
      "xent_avg": c["text_xent"].mean(),
      "xent_std": c["text_xent"].std(),
    })
  
  data.index.names = ["doc_ix","sent_ix"]
  # data.reset_index(level=1,inplace=True)
  feats = data.groupby(level=0).apply(reMergeXents)
  feats = feats.reindex(df.index)

  if context_col == "firstSent":
    log.info("Resetting the sentences")
    df[sent_col] = sents_backup
    
  assert len(feats) == len(df)
  return feats


def main(args):  
  df = loadInput(args)
  print(df.shape)

  # Load tokenizer (GPT3 uses GPT2's tokenizer)
  # will not be needed if the sentences are already there.
  tokenizer = AutoTokenizer.from_pretrained("gpt2")
  
  # Parse into sentences
  df = splitIntoSentences(df,tokenizer,args.story_column,args.sentence_column)
  
  feat_list = {}
  for h in tqdm(args.history_sizes,ascii=True,desc="History sizes"):
    print(file=sys.stderr); sys.stderr.flush()
    feats = computePplxInContext(
      df, args.gpt3_variant, story_col=args.story_column,
      context_col=args.context_column,
      sent_col=args.sentence_column,hist_size=h,
      max_seq_len=MAX_SEQ_LEN)
    feats = feats.rename(columns={c: c+f"_hist{h}" for c in feats})
    feat_list[h] = feats
    print(file=sys.stderr); sys.stderr.flush()
    
  feats = pd.concat(feat_list.values(),axis=1)
  
  df_out = pd.concat([df,feats],axis=1,sort=False)
  if args.output_story_file:
    saveOutput(df_out,args.output_story_file,"stories")
    print(args.output_story_file)
    
  if args.output_sentence_file:
    df_long = meltFeaturesPerSentence(df_out,args,story_level_cols=df.columns.tolist())
    saveOutput(df_long,args.output_sentence_file,"sentences")
    print(args.output_sentence_file)
  log.info("Done")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--input_story_file",help="CSV file with one story per line. "
                      "Use this to analyze multiple stories at once.")
  parser.add_argument("--input_sentence_file",help="CSV file with one sentence per line. "
                      "Use this for one story only.")
  
  parser.add_argument("--output_sentence_file",
                      help="Output file with the linearity of each sentence per line.")
  parser.add_argument("--output_story_file",
                      help="Output file with linearity scores aggregated per story.")
  
  parser.add_argument("--story_column",default="story",
                      help="Column for which to compute the linearity scores")
  
  parser.add_argument("--story_id_column",default="story_id",
                      help="Story identifier (optional, defaults to 'story_id'; "
                      "will assign automatically)")
  parser.add_argument("--sentence_id_column",default="sentence_id",
                      help="Sentence identifier (optional, defaults to 'sentence_id'; "
                      "will assign automatically)")
  
  parser.add_argument("--context_column", default=None,
                      help="Column that contains the context or 'main event'. "
                      "Optional; if ommitted, will use the empty string.")  
  parser.add_argument("--sentence_column",default="sents",
                      help="If text already split up into sentences. "
                      "This column should be a json list of words/tokens.")

  parser.add_argument("--history_sizes",nargs="+",type=int,default=[0, -1])

  parser.add_argument("--debug",type=int,default=0)

  parser.add_argument("--gpt3_variant",default="ada",
                      help="Which GPT3 variant to use: ada, babbage, curie, davinci.")
  
  parser.add_argument("--story_prompt",help="If present, will preprend the story_prompt "
                      "before the story to compute pplx")
  parser.add_argument("--context_prompt",help="If present, will prepend this to the context. "
                      "E.g., Summary: This is about a wedding that I went to last month")
  
  args = parser.parse_args()
  
  if not args.output_story_file and not args.output_sentence_file:
    parser.print_usage()
    print()
    raise ValueError("Please provide an output file: --output_story_file or --output_sentence_file")
  
  log.info(args)
  main(args)

              
