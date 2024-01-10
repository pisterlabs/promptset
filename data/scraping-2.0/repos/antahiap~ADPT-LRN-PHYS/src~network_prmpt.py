from matplotlib import pyplot as plt
import networkx as nx
import pandas as pd
from io import StringIO
from pathlib import Path
import os
import json

# from pyvis.network import Network
from openai_api import OpenAIApi
from network_vis import VisNetwork
from constants import PAPER_PDF_PATH


class NetworkPrmpt():
    def __init__(self, g_data, papers, th):
        self.api = OpenAIApi("gpt-3.5-turbo-16k")
        self.Gd = g_data
        data_name =  '_'.join(papers + [str(th)])
        self.prmpt_path = PAPER_PDF_PATH / Path(f'{data_name}.json')

    def diff_paper(self, n1, n2):

        self.cntnt1 = self._read_from_graph(n1)
        self.cntnt2 = self._read_from_graph(n2)

        get_res = None
        get_res = self.check_exist(n1, n2)
        if not get_res:
            self.sim = self._similaity_papers(n1, n2)
            self.diff = self._differnce_papers(n1, n2)
            
        network = self._to_network()
        return network
        # return None
    

    def _similaity_papers(self, n1, n2):

        prompt= f'''
            Let us consider the following papers as nodes {n1} and {n2}. I want to know how they are similar. For the similarities, give me a list of the similarities, max 5 rows, between these two nodes.
                    
            Output format: in csv format with delimitar="\t" that the header is " dependent paper\t source paper\t importance weight\t content of the relation". The columns content is as follows::

            - col=0: source node as intiger
            - col=1: target node as intiger
            - col=2: importance weight, from 0.5 to 1, 
            - col=3: the content of each relation in max 5 words, do not say both papers or nodes, start with the verb, and don't say the node number.


            Node {n1}, 
            text: {self.cntnt1['text']}

            Node {n2}, 
            text: {self.cntnt2['text']}


        '''

        result, _, _ = self.api.call_api_single(prompt)
        # result ='''dependent paper\tsource paper\timportance weight\tcontent of the relation\n0\t1\t0.9\tTransformer architecture replaces recurrent networks\n0\t1\t0.8\tBLEU score improvement by 2\n0\t1\t0.7\tTransformer achieves 28.4 BLEU on English-to-German translation\n0\t1\t0.6\tTransformer establishes new BLEU score of 41.8 on English-to-French translation\n0\t1\t0.5\tTransformer generalizes well to other tasks'''

        return result

    def _differnce_papers(self, n1, n2):

        prompt= f'''
            Let us consider the following papers as nodes {n1} and {n2}. I want to know what are their main differences.  List the important aspect of each paper that is not mentioned in the other one. a maximum of 4 item for each node {n1} and {n2}.
                    
            Output format: in csv in 3 columns format with delimitar="\t" that the header is "paper\t importance weight\t the important aspect". The columns content is as follows::

            - col=0: the paper as integer
            - col=1: importance weight, from 0.5 to 1, 
            - col=2: the content of difference in max 5 words, do not say both papers or nodes, start with the verb, and don't say the node number.


            Node {n1}, 
            text: {self.cntnt1['text']}

            Node {n2}, 
            text: {self.cntnt2['text']}


        '''
        print('prmpt' ,n1, n2)


        result, _, _ = self.api.call_api_single(prompt)
        # result = '''paper\timportance weight\tthe important aspect\n0\t1\tRNN and CNN not used\n0\t0.8\tMore parallelizable and faster training\n0\t0.7\tHigher BLEU scores in machine translation tasks\n0\t0.6\tSuccessfully applied to English constituency parsing\n1\t1\tNo recurrence in the model architecture\n1\t0.8\tAttention mechanism used for global dependencies\n1\t0.7\tSignificantly more parallelization than recurrent models\n1\t0.6\tState-of-the-art translation quality with shorter training time'''
        return result

    def _read_from_graph(self, n):

        cntnt = {}
        try:
            node = [m for m in self.Gd['nodes'] if m['id'] ==n][0]
        except IndexError:
            print('missing node', n)
            return None
        cntnt['color'] = node['color']
        cntnt['text'] = node['text']
        cntnt['label'] = node['label']
        cntnt['paper'] = node['paper']
        cntnt['ids'] = node['ids']
        cntnt['id'] = node['id']
        cntnt['title'] = node['title']

        return cntnt

    def _to_network(self):

        def trans_color(html_color, alpha):
            alpha = min(255, max(0, alpha))
            html_color = html_color.lstrip('#')

            # Extract the RGB components (assuming it's in #RRGGBB format)
            red = int(html_color[0:2], 16)
            green = int(html_color[2:4], 16)
            blue = int(html_color[4:6], 16)

            rgba_color = f"rgba({red}, {green}, {blue}, {alpha/255:.2f})"

            return rgba_color

        def splt_txt(txt, value):
            words = txt.split()
            lines = []
            current_line = ""

            for word in words:
                if len(current_line) + len(word) + 1 <= value:
                    if current_line:
                        current_line += " "
                    current_line += word
                else:
                    lines.append(current_line)
                    current_line = word

            if current_line:
                lines.append(current_line)

            txt_cut = "\n".join(lines)
            return(txt_cut)

        def _sim_nodes(G):
            try:
                for index, row in df_sim.iterrows(): 

                    info = splt_txt(row[3], 20)
                    w = row[2]

                    if not self.info_id in G.nodes():
                        G.add_node(
                            self.info_id, 
                            color='#B2BEB5', 
                            label=info, 
                            font='18px arial black',
                            size=10)

                    G.add_edge(cn1['id'], self.info_id, weight=w)
                    G.add_edge(self.info_id, cn2['id'], weight=w)

                    self.info_id +=1

            except KeyError:
                print('issue to iterate')
            return G
        
        def _diff_nodes(G, cni, df):
            
            ni = cni['id']
            node_label = f"{cni['paper'][:5]}, {cni['label']}"
            node_title = cni['title']
            G.add_node(
                ni, 
                color=cni['color'], 
                label=node_label, 
                title = node_title, 
                font='20px arial black',
                size=35
                )

            df_i = df[df.iloc[:, 0] == ni]
            for index, row in df_i.iterrows(): 

                info = splt_txt(row[2], 20)
                w = row[1]

                try:
                    color = trans_color(cni['color'], 150)                   
                    G.add_node(
                        self.info_id, 
                        color=color, 
                        label=info, 
                        font='18px arial black',
                        size=10)
                    G.add_edge(ni, self.info_id, weight=w)
                    self.info_id +=1
                except ValueError:
                    continue
            return G
        
        def _sort_res(data):    
            df = pd.read_csv(StringIO('\n'.join(data.split('\n')[1:])), header=None, sep='\t')

            return df

        self.info_id = 100
        G =nx.Graph()

        cn1, cn2  = self.cntnt1, self.cntnt2
        df_diff =  _sort_res(self.diff)
        df_sim =  _sort_res(self.sim)

        G = _diff_nodes(G, cn1, df_diff)
        G = _diff_nodes(G, cn2, df_diff)
        G = _sim_nodes(G)
        
        return G

        layout = nx.spring_layout(G)

        # Draw the graph using Matplotlib
        nx.draw(G, layout, with_labels=True, node_color='skyblue', font_size=10, node_size=500)
        
        plt.show()

    def read_prmpt_file(self):

        if os.path.isfile(self.prmpt_path):
            with open(self.prmpt_path, 'r') as f:
                data = json.load(f)
                return data
        else:
            return None


    def check_exist(self, n1, n2):

        data =self.read_prmpt_file()
        if not data:
            return None
        
        self.key = str((n1, n2))

        if not self.key in data.keys():
            self.key = str((n2, n1))

        if not self.key in data.keys():
            return None

        self.diff = data[self.key]['diff']
        self.sim = data[self.key]['sim']
        return self.key




if __name__ == '__main__':


    src_path = "data/article_pdf/"
    papers = [ "1706.03762"]#, "1312.4400", "1603.06147"]
    


    th = .87
    g = VisNetwork()
    G = g.grph_embd(th, src_path, papers)
    G_data = g.json_network(th, src_path, papers, G=G)
    nt_prmpt = NetworkPrmpt(G_data, papers, th)

    edge_dic = nt_prmpt.read_prmpt_file()
    if not edge_dic:
        edge_dic = {}

    edges = set(G.edges())
    for edge in G.edges():
        src, dst = edge
        check = nt_prmpt.check_exist(src, dst)

        if not check:
           
            nt_prmpt.diff_paper(src, dst)
            edge_dic[str(edge)]={
                'diff':nt_prmpt.diff,
                'sim':nt_prmpt.sim
                }
            # print(edge_dic[str(edge)])
            # break

            with open(nt_prmpt.prmpt_path, 'w' , encoding='utf-8') as f:
                json.dump(edge_dic, f, indent=4) 


