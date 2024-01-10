#!/usr/bin/env python3
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
from tqdm import trange

import torch
import torch.nn.functional as F
import numpy as np

from transformers import GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig, XLMConfig, CTRLConfig

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
from transformers import XLNetLMHeadModel, XLNetTokenizer
from transformers import TransfoXLLMHeadModel, TransfoXLTokenizer
from transformers import CTRLLMHeadModel, CTRLTokenizer
from transformers import XLMWithLMHeadModel, XLMTokenizer


import signal
# found this technique here:
# https://stackoverflow.com/questions/25308847/attaching-a-process-with-pdb
def handle_pdb( sig, frame ):
    import pdb
    pdb.Pdb().set_trace( frame )



logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig, XLMConfig, CTRLConfig)), ())

MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer),
    'ctrl': (CTRLLMHeadModel, CTRLTokenizer),
    'openai-gpt': (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'xlnet': (XLNetLMHeadModel, XLNetTokenizer),
    'transfo-xl': (TransfoXLLMHeadModel, TransfoXLTokenizer),
    'xlm': (XLMWithLMHeadModel, XLMTokenizer),
}

# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PADDING_TEXT = """ In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(model, length, context, num_samples=1, temperature=1, top_k=0, top_p=0.0, repetition_penalty=1.0,
                    is_xlnet=False, is_xlm_mlm=False, xlm_mask_token=None, xlm_lang=None, device='cpu'):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    with torch.no_grad():
        for _ in trange(length):

            inputs = {'input_ids': generated}
            if is_xlnet: 
                # XLNet is a direct (predict same token, not next token) and bi-directional model by default
                # => need one additional dummy token in the input (will be masked), attention mask and target mapping (see model docstring)
                input_ids = torch.cat((generated, torch.zeros((1, 1), dtype=torch.long, device=device)), dim=1)
                perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float, device=device)
                perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token
                target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float, device=device)
                target_mapping[0, 0, -1] = 1.0  # predict last token
                inputs = {'input_ids': input_ids, 'perm_mask': perm_mask, 'target_mapping': target_mapping}

            if is_xlm_mlm and xlm_mask_token:
                # XLM MLM models are direct models (predict same token, not next token)
                # => need one additional dummy token in the input (will be masked and guessed)
                input_ids = torch.cat((generated, torch.full((1, 1), xlm_mask_token, dtype=torch.long, device=device)), dim=1)
                inputs = {'input_ids': input_ids}

            if xlm_lang is not None:
                inputs["langs"] = torch.tensor([xlm_lang] * inputs["input_ids"].shape[1], device=device).view(1, -1)

            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet/CTRL (cached hidden-states)
            next_token_logits = outputs[0][:, -1, :] / (temperature if temperature > 0 else 1.)

            # repetition penalty from CTRL (https://arxiv.org/abs/1909.05858)
            for i in range(num_samples):
                for _ in set(generated[i].tolist()):
                    next_token_logits[i, _] /= repetition_penalty
                
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            if temperature == 0: # greedy sampling:
                next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
    return generated



def getCapPercentage( line ):
    letters = sum( 1 for c in line if c.isalpha() )
    caps = sum( 1 for c in line if c.isupper() )
    return caps / float( letters );
    


def main():
    # if we get a SIGUSR1, we enable the debugger
    signal.signal( signal.SIGUSR1, handle_pdb )

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--out_file", type=str, default="", help="Output file for generated text (triggers infinite mode)" )
    parser.add_argument("--in_file", type=str, default="", help="Input file for prompt text." )
    parser.add_argument("--gen_words", type=str, default="", help="(For infinite mode with --out_file) How many words to generate, max." )
    parser.add_argument("--gen_min_words", type=int, default=0, help="(For infinite mode with --out_file) How many words to generate at minimum." )
    parser.add_argument("--chapter_number", type=int, default=0, help="Prefaces generated text with chapter header, and auto-ends chapter." )
    parser.add_argument("--allow_chapter_sub_headings", type=int, default=0, help="Set to 1 to permit sub-headings under chapters." )
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--padding_text", type=str, default="")
    parser.add_argument("--xlm_lang", type=str, default="", help="Optional language when used with the XLM model.")
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="temperature of 0 implies greedy sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                        help="primarily useful for CTRL model; in that case, use 1.2")
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--stop_token', type=str, default=None,
                        help="Token at which text generation is stopped")
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    set_seed(args)

    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    model.to(args.device)
    model.eval()

    if args.length < 0 and model.config.max_position_embeddings > 0:
        args.length = model.config.max_position_embeddings
    elif 0 < model.config.max_position_embeddings < args.length:
        args.length = model.config.max_position_embeddings  # No generation bigger than model size 
    elif args.length < 0:
        args.length = MAX_LENGTH  # avoid infinite loop

    logger.info(args)
    if args.model_type in ["ctrl"]:
        if args.temperature > 0.7:
            logger.info('CTRL typically works better with lower temperatures (and lower top_k).')

    cumu_text = ""

    keepGoing = True
    
    wordsWritten = 0

    # total text that has been written out to file
    textWrittenOut = ""


    chapterIntroWritten = True

    if args.chapter_number > 0:
        chapterIntroWritten = False

    
    # running list of all text blocks that have been written out to
    # file so far
    textBlocks = [] 
    
    
    # watch out for infinite loops
    maxRewinds = 10
    rewindsSoFar = 0

    rewindJustHappened = False

    # rewinds text written so far by some number of generated blocks
    # rewrites out file entirely to reflect this
    # returns False if we would rewind back too close to the beginning
    # (10 blocks from chapter header, not leaving enough context to continue)
    def rewindBlocks( numToRewind ):
        nonlocal textBlocks, textWrittenOut, cumu_text, rewindJustHappened

        numBlocks = len( textBlocks )
        minBlocks = numToRewind + 10
        if numBlocks <= minBlocks:
            print( "Rewind failed, only " + 
                   str( numBlocks ) + " left, need " + str( minBlocks ) + "\n" )
            return False
        textBlocks = textBlocks[:-numToRewind]
        
        glue = ""
        
        textWrittenOut = glue.join( textBlocks )
        cumu_text = textWrittenOut
        
        # rewrite file
        text_file = open( args.out_file, "w" )
        n = text_file.write( textWrittenOut )
        text_file.close()

        nonlocal wordsWritten

        wordsWritten = textWrittenOut.count( ' ' )
        
        print( "After rewind, final textBlock = '" + 
               textBlocks[-1:][0] + "'\n" )
        
        rewindJustHappened = True
        return True


    while keepGoing:
        xlm_lang = None
        # XLM Language usage detailed in the issues #1414
        if args.model_type in ["xlm"] and hasattr(tokenizer, 'lang2id') and hasattr(model.config, 'use_lang_emb') \
                and model.config.use_lang_emb:
            if args.xlm_lang:
                language = args.xlm_lang
            else:
                language = None
                while language not in tokenizer.lang2id.keys():
                    language = input("Using XLM. Select language in " + str(list(tokenizer.lang2id.keys())) + " >>> ")
            xlm_lang = tokenizer.lang2id[language]

        # XLM masked-language modeling (MLM) models need masked token (see details in sample_sequence)
        is_xlm_mlm = args.model_type in ["xlm"] and 'mlm' in args.model_name_or_path
        if is_xlm_mlm:
            xlm_mask_token = tokenizer.mask_token_id
        else:
            xlm_mask_token = None

        raw_text = ""
        
        if args.in_file:
            if not cumu_text:
                raw_text = open( args.in_file ).read()

                if args.chapter_number > 0:
                    chapterHeader = ( "\n\n\nChapter " + 
                                      str( args.chapter_number ) +
                                      "\n\n" );
                    raw_text = raw_text + chapterHeader 
                             
                    if args.out_file:
                        text_file = open( args.out_file, "a" )
                        n = text_file.write( chapterHeader )
                        textWrittenOut = textWrittenOut + chapterHeader
                        textBlocks.append( chapterHeader )
                        text_file.close()
            else:
                raw_text = cumu_text
        else:
            # we're not in infinite mode
            # keep applying same prompt for next batch, no accumulation
            cumu_text = ""
            if args.prompt:
                raw_text = args.prompt
            else:
                raw_text = input("Model prompt >>> ")

        cumu_text = raw_text

        #print( "Input text has " + str( raw_text.count( ' ' ) ) + " words\n" )

        if args.model_type in ["transfo-xl", "xlnet"]:
            # Models with memory likes to have a long prompt for short inputs.
            raw_text = (args.padding_text if args.padding_text else PADDING_TEXT) + raw_text

        # Problem:  tokenization strips trailing white space
        # but we need that to preserve paragraph structure, etc.
        
        # Solution:  add a dummy token at the end, then remove it later
        # after tokenization is complete.
        endToken = " *"
        context_tokens = tokenizer.encode( raw_text + endToken, 
                                           add_special_tokens=False)

        # now strip it off
        context_tokens = context_tokens[:-1]
        
        
        bufferMax = 1000

        # 1024 is the length limit for context tokens + generated tokens
        # make it 1000 to give us some wiggle room
        if len( context_tokens ) + args.length > bufferMax:
            l = len( context_tokens )

            #print( "Overflow with (" + str( l ) + "tokens)\n" )
            #print( "Old cumu_text = {" + cumu_text + "}\n\n\n" );

            extra = ( l + args.length )  - bufferMax 
            context_tokens = context_tokens[ extra: ]            

            cumu_text = tokenizer.decode( context_tokens, 
                                          clean_up_tokenization_spaces=False )

            #print( "trimmed cumu_text = {" + cumu_text + "}\n\n\n" );
            #l = len( context_tokens )
            #print( "After trimming, have (" + str( l ) + "tokens)\n\n" )
        

        rewindJustHappened = False
            
        #print( "context_tokens (len=" 
        #       + str( len( context_tokens ) ) + ") = " 
        #       + repr( context_tokens ) + "\n" )

        #text_tokens = []
        #
        #for c in context_tokens:
        #    text_tokens.append( 
        #        tokenizer.decode( c, clean_up_tokenization_spaces=False) )
        #
        #print( "text_tokens = " + repr( text_tokens ) + "\n" )
        
        if args.model_type == "ctrl":
            if not any(context_tokens[0] == x for x in tokenizer.control_codes.values()):
                logger.info("WARNING! You are not starting your generation from a control code so you won't get good results")
        out = sample_sequence(
            model=model,
            context=context_tokens,
            num_samples=args.num_samples,
            length=args.length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            is_xlnet=bool(args.model_type == "xlnet"),
            is_xlm_mlm=is_xlm_mlm,
            xlm_mask_token=xlm_mask_token,
            xlm_lang=xlm_lang,
            device=args.device,
        )
        out = out[:, len(context_tokens):].tolist()
        for o in out:
            text = tokenizer.decode(o, clean_up_tokenization_spaces=True)
            
            # this trims off final letter of text if stop_token not found in
            # text.
            #text = text[: text.find(args.stop_token) if args.stop_token else None]

            # how many words in text?
            
            numWords = len( text.split() )

            if( len( text ) > 20 and
                numWords == 1 ) :
                print( "Text '" + 
                       text + "' contains only 1 word, bailing out." )
                keepGoing = False
                break

                
            skipThisText = False

            rewind = False

            # this does the right thing, only trimming if stop_token found
            if args.stop_token :
                loc = text.find(args.stop_token)
                if loc != -1:
                    # actually trimming this block seems to produce
                    # gibberish after, because some nonsense (footnotes)
                    # usually proceeds <|endtext|>
                    # so don't do this:
                    # text = text[:loc]
                    # instead, skip the whole text block
                    # which will effectively regenerate this block again
                    rewind = True
            
            badTokens = [ "@", "http", ".com" ]
            
            for b in badTokens:
                loc = text.find(b)
                if loc != -1:
                    rewind = True

            if rewind:
                rewindsSoFar += 1
                if rewindsSoFar > maxRewinds:
                    print( "Rewound " + str( rewindsSoFar ) +
                           " times, giving up\n" )
                    keepGoing = False
                    break

            if args.out_file and rewind:
                print( "Last block of text, '" + 
                       text + 
                       "' contains rejected token.  Rewinding 10 blocks" )
                if not rewindBlocks( 10 ):
                    print( "Rewind failed, giving up\n" )
                    keepGoing = False
                break
            elif rewind:
                # not in infinite mode, just discard this block and try again
                break
                

            
            if not chapterIntroWritten:
                # make sure first bit of generated text starts with a capital
                # letter or quote.  If not, discard, and try again.
                
                # get rid of beginning white space
                # Chapter header already has right amount of white space
                text = text.lstrip()
                
                if len( text ) == 0:
                    break
                c = text[0]
                if c != "\"" and not c.isupper():
                    # opening with " is okay, or upper case letter
                    print( "Discarding mal-formatted chapter intro: '" +
                           text + "', bad first char '" + c + "'\n" )
                    break
                
                # if we get her, first block passed the test
                chapterIntroWritten = True                
                print( "Found well-formed chapter intro: '" + text + "'\n" )
                

            # now run some basic cleanup on this latest block
            # First, never allow a space after a newline, at the start of a line
            text = text.replace( "\n ", "\n" )
            
            # next, watch for multiple blank lines between paragraphs
            # start with longest examples first
            text = text.replace( "\n\n\n\n\n", "\n\n" )
            text = text.replace( "\n\n\n\n", "\n\n" )
            text = text.replace( "\n\n\n", "\n\n" )
            

            if len( text ) > 0 and text[0] == ' ':
                # consider case where previous block ended with newlines
                # we don't want to start wit spaces here
                if cumu_text.endswith( "\n" ):
                    text = text.lstrip()
            
            # if cumu_text ended with end of paragraph, strip
            # extra lines from this next block
            if cumu_text.endswith( "\n\n" ):
                text = text.lstrip()
            elif cumu_text.endswith( "\n" ):
                # it ended with a single newline
                # this text must start with a single newline
                text = text.lstrip()
                text = "\n" + text



            chapterDone = False

                                    

            if args.out_file:
                text_file = open( args.out_file, "a" )
                n = text_file.write( text )
                textWrittenOut = textWrittenOut + text
                textBlocks.append( text )
                text_file.close()

                wordsWritten = textWrittenOut.count( ' ' )
                
                print( "Generated " + str( wordsWritten ) + " words\n" )
                if args.gen_words:
                    if wordsWritten > int( args.gen_words ):
                        keepGoing = False
                        if args.chapter_number > 0 and not chapterDone:
                            print( "Incompltete chapter ran over word limit\n" )

                # watch for wayward section breaks that weren't caught
                # in a chunk test above (perhaps because they straddled
                # the end of a chunk)
                # if we detect one here, end the chapter (if we've gone
                # long enough) or rewind if we can
                if args.chapter_number > 0 and keepGoing and not chapterDone:
                    lines = textWrittenOut.splitlines()
                    # don't consider last line for this test, since it
                    # may be in-progress
                    lines.pop()
                    lineI = 0
                    for l in lines:
                        # watch for Image caption.
                        captionFound = False
                        loc = text.find( "Image" )
                        if loc != -1:
                            captionFound = True
                        loc = text.find( "IMAGE" )
                        if loc != -1:
                            captionFound = True
                        
                        if captionFound:
                            print( "Found Image caption on line "
                                    + str( lineI ) + 
                                    " ('" + 
                                    l + 
                                    "')\n" )
                            rewindsSoFar += 1
                            if rewindsSoFar > maxRewinds:
                                print( "Rewound " + str( rewindsSoFar ) +
                                       " times, giving up\n" )
                                keepGoing = False
                                break
                            if not rewindBlocks( 10 ):
                                print( "Rewind failed, giving up\n" )
                                keepGoing = False
                                break
                            else:
                                skipThisText = True
                                
                        # don't consider first few lines either
                        # that's our chapter header
                        if( lineI > 3 and 
                            len( l ) >= 1 and 
                            len( l ) < 80 and
                            # not dialog
                            l.find( "\"" ) == -1 and
                            # not a short sentence
                            ( ( not l.endswith( '.' ) and
                                not l.endswith( '?' ) )
                              # or a very short sentence on line 5, even if it's
                              # a sentence
                              or ( len( l ) < 30 and lineI == 5 )
                              # or all (or mostly) caps, even if it's a
                              # sentence
                              or getCapPercentage( l ) > 0.7 ) ):
                            
                            if( lineI == 5 and
                                args.allow_chapter_sub_headings == 1 ):
                                # on line 5, right after our Chapter X
                                # line
                                # and permitted by command-line argument
                                continue
                                
                            print( "Found section header on line "
                                   + str( lineI ) + 
                                   " ('" + 
                                   l + 
                                   "')\n" )
                            
                        
                            if wordsWritten >= args.gen_min_words:
                                print( "Chapter has " + str( wordsWritten ) +
                                       " words, ending before " +
                                       "this found section header\n" )
                                # trim off lines from header onward
                                goodLines = lines[:lineI]
                                glue = "\n"
                                textWrittenOut = glue.join( goodLines )
                                textWrittenOut = textWrittenOut.rstrip()
                            
                                textWrittenOut = ( textWrittenOut + 
                                                   "\n\nEND OF CHAPTER" )
                                    
                                # rewrite entire file
                                text_file = open( args.out_file, "w" )
                                n = text_file.write( textWrittenOut )
                                text_file.close()
                            
                                wordsWritten = textWrittenOut.count( ' ' )
                                chapterDone = True
                                break
                            else:
                                rewindsSoFar += 1
                                if rewindsSoFar > maxRewinds:
                                    print( "Rewound " + str( rewindsSoFar ) +
                                           " times, giving up\n" )
                                    keepGoing = False
                                    break
                                print( "Chapter only has " + 
                                       str( wordsWritten ) +
                                       " words, rewinding back before section"
                                       + "header to try making it longer\n" )
                                if not rewindBlocks( 10 ):
                                    print( "Rewind failed, giving up\n" )
                                    keepGoing = False
                                    break
                                else:
                                    skipThisText = True
                                
                        lineI += 1
                    
            else:
                print(text)
                if args.num_samples > 1:
                    print( "[END_SAMPLE]\n\n" )
                    skipThisText = True
                    textWrittenOut = textWrittenOut + text
                    wordsWritten = textWrittenOut.count( ' ' )
                    
                    if wordsWritten >= args.gen_min_words:
                        keepGoing = False

            if chapterDone:
                print( "Compltete chapter with " + 
                       str( wordsWritten ) + " words\n" )
                keepGoing = False;

            if not skipThisText:
                cumu_text = cumu_text + text
                

        if args.prompt and not ( args.out_file or args.gen_min_words > 0 ):
            break
    return text


if __name__ == '__main__':
    main()
