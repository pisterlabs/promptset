#!/usr/bin/env python3
# coding: utf-8


# Software Name: UDParse
# SPDX-FileCopyrightText: Copyright (c) 2021 Orange
# SPDX-License-Identifier: Mozilla Public License 2.0
#
# This software is distributed under the MPL-2.0 license.
# the text of which is available at https://www.mozilla.org/en-US/MPL/2.0/
# or see the "LICENSE" file for more details.
#
# Author: Johannes HEINECKE <johannes(dot)heinecke(at)orange(dot)com> et al.


import collections
import logging
import os
import psutil
import regex
import socket
import sys
import time
import requests

# do not show FutureWarnings from numpy
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import torch

import UDParse.progClient as progClient

# from transformers import XLMModel, XLMTokenizer
from transformers import GPT2Model, GPT2Tokenizer
from transformers import OpenAIGPTModel, OpenAIGPTTokenizer
from transformers import BertModel, BertTokenizer
from transformers import RobertaModel, RobertaTokenizer
from transformers import XLMRobertaModel, XLMRobertaTokenizer
from transformers import CamembertModel, CamembertTokenizer
from transformers import DistilBertModel, DistilBertTokenizer
from transformers import FlaubertModel, FlaubertTokenizer
from transformers import MT5Model, MT5Tokenizer
from transformers import T5Model, T5Tokenizer



import UDParse.conllustats

# ideas from  https://gitlab.tech.orange/nlp/onnx
#from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions

logger = logging.getLogger("udparse")


#from tensorflow.keras import mixed_precision
#policy = mixed_precision.Policy('mixed_float16')
#mixed_precision.set_global_policy(policy)

class Embeddings:
    def __init__(self, lg, progServer=None, gpu=-1):
        logger.warning("Using Pytorch for vectorisation " + __name__)
        logger.warning("Pytorch Version: %s on CUDA %s " % (torch.version.__version__, torch.version.cuda))
        self.lg = lg
        self.gpu = gpu  # GPU device used
        self.progressServer = progServer
        self.netlength = 512
        self.re_extras = regex.compile(r"^\d+-|^\d+\.")
        if self.progressServer:
            logger.info("logging progress on %s" % self.progressServer)

        logger.warning("Numpy version: %s" % np.version.full_version)

        self.display_one_of_X = 1
        if torch.cuda.is_available():
            print("CUDA is available for Pytorch")
            self.display_one_of_X = 7
            logger.warning("using GPU device '%s'" % os.environ.get("CUDA_VISIBLE_DEVICES"))
            self.cuda_available = True
        else:
            self.cuda_available = False

        logger.info("using '%s' to create embeddings" % lg)
        self.use_onnx = False

        if lg == "multi" or lg == "bert":
            #if self.use_onnx:
            #    # ONNX
            #    from transformers import AutoTokenizer
            #    print("============================== onnx")
            #    self.tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased", use_fast=True)
            #    options = SessionOptions()
            #    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
            #    self.model = InferenceSession("/home/xxxx/tools/onnx/bert-base-multilingual-cased.onnx", options, providers=["CPUExecutionProvider"])
            #else:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
            self.model = BertModel.from_pretrained("bert-base-multilingual-cased")

            # else:
            #    # https://github.com/huggingface/transformers/issues/677
            #    pathname = MODELDIR + "/bert-base-multilingual-cased"
            #    self.tokenizer = BertTokenizer.from_pretrained(pathname + "/bert-base-multilingual-cased-vocab.txt", do_lower_case=False)
            #    self.model = BertModel.from_pretrained(pathname)

            if self.cuda_available:
                # print(dir(self.tokenizer))
                self.model = self.model.cuda()
                
            self.getVectors = self.getVectorsBert
        elif lg == "bert-uncased":
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-uncased")
            self.model = BertModel.from_pretrained("bert-base-multilingual-uncased")
            if self.cuda_available:
                self.model = self.model.cuda()
            self.getVectors = self.getVectorsBert

        elif lg == "electra":
            from transformers import ElectraModel, ElectraTokenizer
            self.tokenizer = ElectraTokenizer.from_pretrained("google/electra-base-discriminator")
            self.model = ElectraModel.from_pretrained("google/electra-base-discriminator")
            if self.cuda_available:
                self.model = self.model.cuda()
            self.getVectors = self.getVectorsBert

        elif lg == "electra_fr":
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained("dbmdz/electra-base-french-europeana-cased-discriminator")
            self.model = AutoModel.from_pretrained("dbmdz/electra-base-french-europeana-cased-discriminator")
            if self.cuda_available:
                self.model = self.model.cuda()
            self.getVectors = self.getVectorsBert

        elif lg == "distilbert":
            self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-multilingual-cased")
            self.model = DistilBertModel.from_pretrained("distilbert-base-multilingual-cased")
            if self.cuda_available:
                self.model = self.model.cuda()
            self.getVectors = self.getVectorsBert
        elif lg == "extremdistilbert":
            self.tokenizer = BertTokenizer.from_pretrained("microsoft/xtremedistil-l6-h256-uncased")
            self.model = BertModel.from_pretrained("microsoft/xtremedistil-l6-h256-uncased")
            if self.cuda_available:
                self.model = self.model.cuda()
            self.getVectors = self.getVectorsBert

        elif lg == "itBERT":
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-italian-xxl-cased")
            self.model = AutoModel.from_pretrained("dbmdz/bert-base-italian-xxl-cased")
            if self.cuda_available:
                self.model = self.model.cuda()

            self.getVectors = self.getVectorsBert

        elif lg == "gaBERT":
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained("DCU-NLP/bert-base-irish-cased-v1")
            self.model = AutoModel.from_pretrained("DCU-NLP/bert-base-irish-cased-v1")
            if self.cuda_available:
                self.model = self.model.cuda()

            self.getVectors = self.getVectorsBert

        elif lg == "arBERT":
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained("asafaya/bert-base-arabic")
            self.model = AutoModel.from_pretrained("asafaya/bert-base-arabic")

            if self.cuda_available:
                self.model = self.model.cuda()

            self.getVectors = self.getVectorsBert

        elif lg == "fiBERT":
            from transformers import AutoTokenizer, AutoModel  # WithLMHead

            self.tokenizer = AutoTokenizer.from_pretrained("TurkuNLP/bert-base-finnish-cased-v1")
            self.model = AutoModel.from_pretrained("TurkuNLP/bert-base-finnish-cased-v1")

            if self.cuda_available:
                self.model = self.model.cuda()

            self.getVectors = self.getVectorsBert

        elif lg == "slavicBERT":
            logging.warning("only available for pytorch, install pytorch in your environment")
            from transformers import AutoTokenizer, AutoModel  # WithLMHead

            self.tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/bert-base-bg-cs-pl-ru-cased")
            self.model = AutoModel.from_pretrained("DeepPavlov/bert-base-bg-cs-pl-ru-cased")

            if self.cuda_available:
                self.model = self.model.cuda()

            self.getVectors = self.getVectorsBert

        elif lg == "plBERT":
            logging.warning("only available for pytorch, install pytorch in your environment")
            from transformers import AutoTokenizer, AutoModel  # WithLMHead

            self.tokenizer = AutoTokenizer.from_pretrained("dkleczek/bert-base-polish-uncased-v1")
            self.model = AutoModel.from_pretrained("dkleczek/bert-base-polish-uncased-v1")

            if self.cuda_available:
                self.model = self.model.cuda()

            self.getVectors = self.getVectorsBert

        elif lg == "svBERT":
            from transformers import AutoTokenizer, AutoModel  # WithLMHead

            self.tokenizer = AutoTokenizer.from_pretrained("KB/bert-base-swedish-cased")
            self.model = AutoModel.from_pretrained("KB/bert-base-swedish-cased")

            if self.cuda_available:
                self.model = self.model.cuda()

            self.getVectors = self.getVectorsBert

        elif lg == "nlBERT":
            from transformers import AutoTokenizer, AutoModel  # WithLMHead

            self.tokenizer = AutoTokenizer.from_pretrained("wietsedv/bert-base-dutch-cased")
            self.model = AutoModel.from_pretrained("wietsedv/bert-base-dutch-cased")

            if self.cuda_available:
                self.model = self.model.cuda()

            self.getVectors = self.getVectorsBert

        elif lg == "flaubert":
            # https://github.com/getalp/Flaubert huggingface: flaubert/flaubert_base_cased
            # wget https://zenodo.org/record/3567594/files/xlm_bert_fra_base_lower.tar
            # useold = False
            # if useold:
            #    modelname = MODELDIR + "/xlm_bert_fra_base_lower"
            #    #print("zzzz", modelname)
            #    self.model, log = XLMModel.from_pretrained(modelname, output_loading_info=True)
            #    ##print(log)
            #
            #    ## Load tokenizer
            #    self.tokenizer = XLMTokenizer.from_pretrained(modelname, do_lowercase_and_remove_accent=False)
            # else:
            logging.warning("only available for pytorch")
            self.tokenizer = FlaubertTokenizer.from_pretrained("flaubert/flaubert_base_cased")
            self.model = FlaubertModel.from_pretrained("flaubert/flaubert_base_cased")

            if self.cuda_available:
                self.model = self.model.cuda()

            self.getVectors = self.getVectorsBert

        elif lg == "camembert" or lg == "fr":
            self.model = CamembertModel.from_pretrained("camembert-base")
            self.tokenizer = CamembertTokenizer.from_pretrained("camembert-base", do_lower_case=False)

            if self.cuda_available:
                self.model = self.model.cuda()

            self.getVectors = self.getVectorsBert

        elif lg == "roberta" or lg == "en":
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-large", do_lower_case=False)
            self.model = RobertaModel.from_pretrained("roberta-large")

            if self.cuda_available:
                self.model = self.model.cuda()

            self.getVectors = self.getVectorsBert

        elif lg == "mt5":
            self.tokenizer = MT5Tokenizer.from_pretrained("google/mt5-large", do_lower_case=False)
            self.model = MT5Model.from_pretrained("google/mt5-large")

            if self.cuda_available:
                self.model = self.model.cuda()

            self.getVectors = self.getVectorsBert

        elif lg == "t5":
            self.tokenizer = T5Tokenizer.from_pretrained("t5-large", do_lower_case=False)
            self.model = T5Model.from_pretrained("t5-large")

            if self.cuda_available:
                self.model = self.model.cuda()

            self.getVectors = self.getVectorsBert

        elif lg == "xml-roberta" or lg == "xlmr":
            #print("SSSSSSS xlmr")
            #from transformers import XLMRobertaModel, XLMRobertaTokenizer
            self.tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-large", do_lower_case=False)
            self.model = XLMRobertaModel.from_pretrained("xlm-roberta-large")

            if self.cuda_available:
                self.model = self.model.cuda() #, jit_compile=True)

            self.getVectors = self.getVectorsBert

        elif lg == "gpt":
            self.tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt", do_lower_case=False)
            # DOES NOT WORK
            # print("CLS", self.tokenizer.cls_token)
            # if not self.tokenizer.cls_token:
            #    self.tokenizer.add_special_tokens({'cls_token': '[CLS]'})
            #    self.tokenizer.add_special_tokens({'sep_token': '[SEP]'})

            self.model = OpenAIGPTModel.from_pretrained("openai-gpt")

            if self.cuda_available:
                self.model = self.model.cuda()

            self.getVectors = self.getVectorsBert

        elif lg in ["gpt2", "gpt2-medium", "gpt2-large"]:
            self.tokenizer = GPT2Tokenizer.from_pretrained(lg, do_lower_case=False)
            # DOES NOT WORK
            # if not self.tokenizer.cls_token:
            #    self.tokenizer.add_special_tokens({'cls_token': '[CLS]'})
            #    self.tokenizer.add_special_tokens({'sep_token': '[SEP]'})

            self.model = GPT2Model.from_pretrained(lg)

            if self.cuda_available:
                # print(dir(self.tokenizer))
                self.model = self.model.cuda()

            self.getVectors = self.getVectorsBert

        self.model.eval()
        self.ctlong = 0
        logging.info("embeddings '%s' read" % lg)

    def filterOS(self, t):  # t is a 1-dim torch tensor of sentencepiece indexes
        condition = t != self.bos
        t = t[condition]
        condition = t != self.eos
        t = t[condition]
        return t

    def getVectorsBert(self, sentence):  # [[cols]]
        ##aa = time.time()
        # concatenate tokens to normal sentence
        # text_with_mwt = ""
        words = []
        for cols in sentence:
            # if "-" in cols[0]:
            if self.re_extras.match(cols[0]):
                continue
            # text_with_mwt += cols[1]
            words.append(cols[1])

        # print("=========cls:", self.tokenizer.cls_token)
        # print("=========sep:", self.tokenizer.sep_token)
        # add CLS and SEP tokens
        if self.tokenizer.cls_token:
            words = [self.tokenizer.cls_token, *words, self.tokenizer.sep_token]  # add [CLS] and [SEP] around sentence
        else:
            # GPT2 has no CLS/SEP tokens
            words = [*words]  # add nothing



        #print("AAA", len(words))
        # print("BBB", self.tokenizer.encode(words[0], None, add_special_tokens=False))
        # TODO: optimiser
        tokens = [self.tokenizer.encode(t, None, add_special_tokens=False) for t in words]

        #print("words", words)
        #print("token", len(tokens))
        #tokens2 = self.tokenizer.encode(words, None, add_special_tokens=False, is_split_into_words=True)

        # there are at times "words" which only consists of strange codepoints
        # the tokenizer creates an empty list. So we put the code for "." token instead
        for tt in tokens:
            if len(tt) == 0:
                tt.append(119)  # "."

        #if True: #False:  # debug:
        #     # print("TOKEN", tokens)
        #     for tt in tokens:
        #         # print("TOKS", len(tt), end=" ")
        #         for t in tt:
        #             print("%d %d/<%s>" % (len(self.tokenizer.decode([t])), t, self.tokenizer.decode([t])), end=" ")
        #         print()
        #     print()
        #     for tt in tokens2:
        #         # print("TOKS", len(tt), end=" ")
        #         for t in [tt]:
        #             print("%d %d/<%s>" % (len(self.tokenizer.decode([t])), t, self.tokenizer.decode([t])), end=" ")
        #         print()

        #bb = time.time()
        lengths = [len(w) for w in tokens]

        tokens = [t for s in tokens for t in s]  # flatten tokens list
        #print("tokensflat", tokens)

        tlen = len(tokens)
        number_of_parts = ((tlen - 1) // self.netlength) + 1
        if number_of_parts > 1:
            logging.warning("long sentence %d          " % tlen)

        # unpadded = tokens[:]
        #attention = [1] * len(tokens)
        # PAdding
        #if tlen < self.netlength:
        #    # padding
        #    padlength = (number_of_parts * self.netlength) - tlen
        #    if self.tokenizer.pad_token_id == None:
        #        tokens += [0] * padlength
        #        #attention += [0] * padlength
        #    else:
        #        # tokens += [self.tokenizer.pad_token_id] * ((number_of_parts * self.netlength)-tlen)
        #        tokens += [self.tokenizer.pad_token_id] * padlength
        #        #attention += [0] * padlength


        #if self.use_onnx:
        #    token_tensors = tokens
        #    attention_tensor = attention
        #else:
        token_tensors=torch.tensor([tokens])
        #attention_tensor=torch.tensor([attention])
        #print("ZZZZneu", len(tokens), type(token_tensors), token_tensors.shape)

        llfeats = []
        #print("number", number_of_parts)
        with torch.no_grad():
            for p in range(number_of_parts):
                #if self.use_onnx:
                #    currenttokens = token_tensors[p * self.netlength : (p + 1) * self.netlength]
                #    currentattention = attention_tensor[p * self.netlength : (p + 1) * self.netlength]
                #else:
                currenttokens = token_tensors[:, p * self.netlength : (p + 1) * self.netlength]
                #currentattention = attention_tensor[:, p * self.netlength : (p + 1) * self.netlength]

                #print("ooooo", currenttokens)
                #print("OOOOO", currentattention)
                #if self.use_onnx:
                #    #print("ffff", currenttokens)
                #    last_layer_features = self.model.run(None, {"input_ids": [currenttokens], "attention_mask": [currentattention]})[0]
                #else:
                # T5, mT5
                if "t5" in self.lg:
                    print("NOT YET IMPLEMENTED")
                    pass
                    #last_layer_features = self.model.encoder(currenttokens, attention_mask=currentattention, training=False)[0]
                else:
                    if self.cuda_available:
                        last_layer_features = self.model(currenttokens.cuda()#, attention_mask=currentattention.cuda()
                                                         )[0]
                    else:
                        last_layer_features = self.model(currenttokens#, attention_mask=currentattention
                                                         )[0]

                llfeats.append(last_layer_features)

        last_layer_features = torch.cat(llfeats, 1)
        #cc = time.time()

        ci = 0  # sentencepiece index to extract
        vectors = []  # list of mean vectors for words

        for wl in range(len(lengths)):
            cw = []  # current word vectors
            # print("----", wl)
            for _ in range(lengths[wl]):
                # print(ci, len(cw), last_layer_features.shape)
                cw.append(last_layer_features[0, ci, :])
                ci += 1
            mean = torch.mean(torch.stack(cw), dim=0)
            #mean = tf.reduce_mean(tf.stack(cw), axis=0)
            # print("ZZZZ", mean.shape)
            #vectors.append(mean)  # .numpy())
            vectors.append(mean.detach().cpu().numpy())

        #dd = time.time()
        #print("INTERNAL_VECTORISATION_PARSING:toktime:enctime:jointime", bb-aa, cc-bb, dd-cc)
        #for vct in vectors:
        #    vv = "%s" % vct
        #    print(vv.replace("\n", " "))

        if self.tokenizer.cls_token:
            return vectors[1 : len(vectors) - 1]  # for the server
        else:
            return vectors


    def process(self, outfile, fns):
        logging.info("creating outdir %s" % os.path.dirname(outfile))
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        
        self.vectors = collections.OrderedDict()  # sentencenum: [vectors]

        block = []  # list of conllu-lines

        # senttext = None
        aa = time.time()
        for fn in fns:
            progclient = None
            if self.progressServer:
                progclient = progClient.PS_Client("UDParse Embeddings", self.progressServer)
            logging.info("reading from %s" % fn)
            cc = UDParse.conllustats.CountConllu(fn)
            logging.info("  %d sentences, %d words" % (cc.sct, cc.wct))

            ifp = open(fn)

            # self.progress(fn, 0, cc.sct)
            if progclient:
                progclient.update(
                    index=psutil.Process().pid,
                    type=self.lg,
                    gpu="%s:%s" % (socket.gethostname(), self.gpu),
                    filename=fn,
                    sentences="%d/%d" % (0, cc.sct),
                )
            for line in ifp:
                line = line.strip()
                if not line:
                    # process sentence
                    sid = len(self.vectors)  # count sentences by counting vectors (there is one vector per sentence)

                    # print(block)
                    if sid % self.display_one_of_X == 0:
                        print("sentence %6d/%d" % (sid, cc.sct), end="\r")
                        if progclient and sid % 28 == 1:
                            progclient.update(
                                index=psutil.Process().pid,
                                type=self.lg,
                                gpu="%s:%s" % (socket.gethostname(), self.gpu),
                                filename=fn,
                                sentences="%d/%d" % (sid, cc.sct),
                            )

                    self.vectors["%s" % sid] = self.getVectors(block)

                    block = []

                else:
                    # if line.startswith("# text ="):
                    #    senttext = line[9:].strip()
                    if line[0] == "#":
                        continue
                    else:
                        elems = line.split("\t")
                        block.append(elems)
            if block:
                sid = len(self.vectors)
                self.vectors["%s" % sid] = self.getVectors(block)

            # print(len(self.vectors), "                  ")
            progclient = None

        # print(len(self.vectors), "                  ")
        bb = time.time()
        print("%d sentences processed in %d secs" % (cc.sct, bb-aa))
        logging.warning("number of long sentences: %d" % self.ctlong)
        # print(self.vectors)

        # create npz
        #os.makedirs(os.path.dirname(outfile), exist_ok=True)
        np.savez(outfile, **self.vectors)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--lg", default="multi", type=str, help="Language (fr/en)")
    parser.add_argument("--outfile", "-o", required=True, help="output filename")
    parser.add_argument("--infiles", "-i", required=True, nargs="+", help="output filename")

    if len(sys.argv) < 1:
        parser.print_help()
    else:
        args = parser.parse_args()

    cc = Embeddings(lg=args.lg)

    cc.process(args.outfile, args.infiles)
