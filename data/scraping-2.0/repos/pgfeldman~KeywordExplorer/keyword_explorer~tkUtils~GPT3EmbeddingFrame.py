import re
import numpy as np
import tkinter as tk
from tkinter import ttk
import tkinter.messagebox as message
from datetime import datetime
from tkinter import filedialog

import pandas as pd
from matplotlib import pyplot as plt
import openai.embeddings_utils as oaiu

from keyword_explorer.tkUtils.ConsoleDprint import ConsoleDprint
from keyword_explorer.tkUtils.Buttons import Buttons
from keyword_explorer.tkUtils.ToolTip import ToolTip
from keyword_explorer.tkUtils.TextField import TextField
from keyword_explorer.tkUtils.DataField import DataField
from keyword_explorer.tkUtils.TopicComboExt import TopicComboExt
from keyword_explorer.OpenAI.OpenAIComms import OpenAIComms
from keyword_explorer.tkUtils.LabeledParam import LabeledParam
from keyword_explorer.utils.ManifoldReduction import ManifoldReduction, EmbeddedText, ClusterInfo
from keyword_explorer.utils.SharedObjects import SharedObjects

from typing import List, Dict, Callable

class GPT3EmbeddingSettings:
    pca_dim:int
    eps:float
    min_samples:int
    perplexity:float
    model:str

    def __init__(self, pca_dim = 5, eps = 64.0, min_samples = 4, perplexity = 2,
                 model = 'text-embedding-ada-002'):
        self.pca_dim = pca_dim
        self.eps = eps
        self.min_samples = min_samples
        self.perplexity = perplexity
        self.model = model

    def from_dict(self, d:Dict):
        if 'PCA_dimensions' in d:
            self.pca_dim = d['PCA_dimensions']
        if 'EPS' in d:
            self.eps = d['EPS']
        if 'min_samples' in d:
            self.min_samples = d['min_samples']
        if 'perplexity' in d:
            self.perplexity = d['perplexity']
        if 'embedding_model' in d:
            self.model = d['embedding_model']


class GPT3EmbeddingFrame:
    oai: OpenAIComms
    mr: ManifoldReduction
    so:SharedObjects
    dp:ConsoleDprint
    generate_model_combo: TopicComboExt
    prompt_text_field:TextField
    response_text_field:TextField
    tokens_param: LabeledParam
    temp_param: LabeledParam
    presence_param: LabeledParam
    frequency_param: LabeledParam
    regex_field:DataField
    auto_field:DataField
    buttons:Buttons
    saved_prompt_text:str
    saved_response_text:str

    def __init__(self, oai:OpenAIComms, mr:ManifoldReduction, dp:ConsoleDprint, so:SharedObjects):
        self.oai = oai
        self.dp = dp
        self.mr = mr
        self.so = so

    def build_frame(self, frm: ttk.Frame, text_width:int, label_width:int):
        engine_list = self.oai.list_models(keep_list = ["embedding"])
        row = 0
        self.embed_model_combo = TopicComboExt(frm, row, "Engine:", self.dp, entry_width=25, combo_width=25)
        self.embed_model_combo.set_combo_list(engine_list)
        self.embed_model_combo.set_text(engine_list[0])
        self.embed_model_combo.tk_combo.current(0)
        row = self.embed_model_combo.get_next_row()
        row = self.build_embed_params(frm, row)
        self.embed_state_text_field = TextField(frm, row, "Embed state:", text_width, height=10, label_width=label_width)
        ToolTip(self.embed_state_text_field.tk_text, "Embedding progess")
        row = self.embed_state_text_field.get_next_row()
        self.buttons = Buttons(frm, row, "Commands", label_width=10)
        b = self.buttons.add_button("Reduce", self.reduce_dimensions_callback, -1)
        ToolTip(b, "Reduce to 2 dimensions with PCS and TSNE")
        b = self.buttons.add_button("Cluster", self.cluster_callback, -1)
        ToolTip(b, "Compute clusters on reduced data")
        b = self.buttons.add_button("Plot", self.plot_callback, -1)
        ToolTip(b, "Plot the clustered points using PyPlot")
        b = self.buttons.add_button("Topics", self.topic_callback, -1)
        ToolTip(b, "Use GPT to guess at topic names for clusters")
        row = self.buttons.get_next_row()

    def add_button(self, label:str, callback:Callable, tooltip:str):
        b = self.buttons.add_button(label, callback)
        ToolTip(b, tooltip)

    def build_embed_params(self, parent:tk.Frame, row:int) -> int:
        f = tk.Frame(parent)
        f.grid(row=row, column=0, columnspan=2, sticky="nsew", padx=1, pady=1)
        self.pca_dim_param = LabeledParam(f, 0, "PCA Dim:")
        self.pca_dim_param.set_text('10')
        ToolTip(self.pca_dim_param.tk_entry, "The number of dimensions that the PCA\nwill reduce the original vectors to")
        self.eps_param = LabeledParam(f, 2, "EPS:")
        self.eps_param.set_text('8')
        ToolTip(self.eps_param.tk_entry, "DBSCAN: Specifies how close points should be to each other to be considered a part of a \ncluster. It means that if the distance between two points is lower or equal to \nthis value (eps), these points are considered neighbors.")
        self.min_samples_param = LabeledParam(f, 4, "Min Samples:")
        self.min_samples_param.set_text('5')
        ToolTip(self.min_samples_param.tk_entry, "DBSCAN: The minimum number of points to form a dense region. For \nexample, if we set the minPoints parameter as 5, then we need at least 5 points \nto form a dense region.")
        self.perplexity_param = LabeledParam(f, 6, "Perplex:")
        self.perplexity_param.set_text('80')
        ToolTip(self.perplexity_param.tk_entry, "T-SNE: The size of the neighborhood around each point that \nthe embedding attempts to preserve")
        return row + 1

    def set_params(self, settings:GPT3EmbeddingSettings):
        self.embed_model_combo.clear()
        self.embed_model_combo.set_text(settings.model)
        self.pca_dim_param.set_text(str(settings.pca_dim))
        self.eps_param.set_text(str(settings.eps))
        self.min_samples_param.set_text(str(settings.min_samples))
        self.perplexity_param.set_text(str(settings.perplexity))

    def get_settings(self) -> GPT3EmbeddingSettings:
        gs = GPT3EmbeddingSettings()
        gs.model = self.embed_model_combo.get_text()
        gs.eps = self.eps_param.get_as_float()
        gs.pca_dim = self.pca_dim_param.get_as_int()
        gs.min_samples = self.min_samples_param.get_as_int()
        gs.perplexity = self.perplexity_param.get_as_int()
        return gs

    def reduce_dimensions_callback(self):
        rf:DataField

        pca_dim = self.pca_dim_param.get_as_int()
        perplexity = self.perplexity_param.get_as_int()
        self.embed_state_text_field.add_text("Reducing: PCA dim = {}  perplexity = {}".format(pca_dim, perplexity))
        self.mr.calc_embeding(perplexity=perplexity, pca_components=pca_dim)
        rf = self.so.get_object("reduced_field")
        if rf != None:
            rf.set_text(len(self.mr.embedding_list))
        print("\tFinished dimension reduction")
        message.showinfo("reduce_dimensions_callback", "Reduced to {} dimensions".format(pca_dim))

    def cluster_callback(self):
        print("Clustering")
        cf:DataField

        eps = self.eps_param.get_as_float()
        min_samples = self.min_samples_param.get_as_int()
        self.mr.dbscan(eps=eps, min_samples=min_samples)
        self.mr.calc_clusters()
        cf = self.so.get_object("clusters_field")
        if cf != None:
            cf.set_text(str(len(self.mr.embedding_list)))

    def topic_callback(self):
        ci:ClusterInfo
        et:EmbeddedText
        split_regex = re.compile("\d+\)")

        for ci in self.mr.cluster_list:
            et_list = []
            for et in ci.member_list:
                et_list.append(et.to_dict())
            df = pd.DataFrame(et_list)
            mean_embedding = list(df['reduced'].mean())
            # Get the distances from the embeddings
            df['distances'] = oaiu.distances_from_embeddings(mean_embedding, list(df['reduced'].values), distance_metric='cosine')
            df2 = df.sort_values('distances', ascending=True)
            text_list = []
            for i, row in df2.iterrows():
                text = str(row['text'])
                text_list.append(text)
                if len(text_list) > 5:
                    break

            prompt = "Extract keywords from this text:\n\n{}\n\nTop three keywords\n1)".format(" ".join(text_list))
            # print("\nCluster ID {} query text:\n{}".format(ci.id, prompt))
            result = self.oai.get_prompt_result_params(prompt, temperature=0.5, max_tokens=60, top_p=1.0, frequency_penalty=0.8, presence_penalty=0)
            l = split_regex.split(result)
            response = "".join(l)
            ci.label = "[{}] {}".format(ci.id, response)
            print("Cluster {}: = {}".format(ci.id, response))
            for et in ci.member_list:
                et.cluster_name = response
        message.showinfo("topic_callback", "Generated {} topics".format(len(self.mr.cluster_list)))

    def plot_callback(self):
        print("Plotting")
        ef:DataField
        title = "Unset Title"
        ef = self.so.get_object("experiment_field")
        if ef != None:
            title = ef.get_text()
        perplexity = self.perplexity_param.get_as_int()
        eps = self.eps_param.get_as_int()
        min_samples = self.min_samples_param.get_as_int()
        pca_dim = self.pca_dim_param.get_as_int()
        self.mr.plot("{}\ndim: {}, eps: {}, min_sample: {}, perplex = {}".format(
            title, pca_dim, eps, min_samples, perplexity))
        plt.show()
