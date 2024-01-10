import openai
import pandas as pd
import spacy
from flashtext import KeywordProcessor
import nltk
from langchain.llms import OpenAI
import os
from dotenv import load_dotenv, find_dotenv

pathprefix = r"D:\Code Working Area\Jupyter\knowledge-graph-for-stakeholder-risks-detection-in-mega-infrastructure-projects\ExcelData"
openai.api_key="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"
os.environ["OPENAI_API_KEY"] = "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"
load_dotenv(find_dotenv())

class File_Path:
    jsfile = pathprefix + "\\Transactions.json"
    stop_words = pathprefix + "\\stop_words\\stop_words_stakeholder.txt"
    unknow_long_phrase = pathprefix + "\\stop_words\\unknow_long.txt"

    project_sor = pathprefix + "\\Source\\project.csv"
    risk_sor = pathprefix + "\\Source\\risk.xlsx"
    stake_sor = pathprefix + "\\Source\\stakeholder.csv"

    project_key = pathprefix + "\\project_keyword\\Project_keyword.xlsx"
    risk_key_path = pathprefix + "\\risk_keyword\\Risk_keyword.xlsx"
    stake_key = pathprefix + "\\stakeholder_keyword\\third_layer_iteration_one_stakeholder.xlsx"

    def read_source(self):
        self.prj_sor = pd.read_csv(self.project_sor, sep=",")["Article Title"].dropna(inplace=True, how="any")
        self.risk_sor = pd.read_excel(self.risk_sor)["Abstract"].dropna(inplace=True, how="any")
        self.stk_sor = pd.read_csv(self.stake_sor, sep=",")["Abstract"].dropna(inplace=True, how="any")

    def read_keyword(self):
        self.prj_key = pd.read_excel(self.project_key, index_col=None)
        self.risk_key = pd.read_excel(self.risk_key_path, index_col=None)
        self.stk_key = pd.read_excel(self.stake_key, index_col=None)
        self.stk_key.dropna(inplace=True)
        self.stk_key.Abstract = self.stk_key.Abstract.str.lower()

        # Do I really need below two steps?
        self.risk_key.Abstract = self.risk_key.Abstract.astype(str)
        self.stk_key.Abstract = self.stk_key.Abstract.astype(str)

    def spacy_init(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.stopwords = self.nlp.Defaults.stop_words


class Filtering(File_Path):
    def __init__(self, dataset: pd.DataFrame):
        super().__init__()
        self.df=dataset
        # self.read_source()
        self.read_keyword()
        self.spacy_init()
        # do this for keyword abstract
        self.prjkey_list = self.prj_key["Article"].to_list()
        self.riskey_list = self.risk_key["Abstract"].to_list()
        self.stkey_list = self.stk_key.Abstract.to_list()

    def keyprocess(self):
        """initialize keywordprocessor class"""
        self.keypro = KeywordProcessor()
        self.keypro.add_keywords_from_list(self.prjkey_list + self.riskey_list + self.stkey_list)

    def abst(self, args: str):
        return self.keypro.extract_keywords(args)

    def pre_suffix_strip(self, ffix: str, df: pd.DataFrame):
        if ffix=="prj":
            df["word"]=df.words.apply(lambda x: x.replace(" project", ""))
        elif ffix=="risk":
            df["word"]=df.words.apply(lambda x: x.replace(" risk", ""))
        return df

    def keyword_identify(self, dataset):
        prj_mask, risk_mask=dataset.words.apply(lambda x: "project" in x[-8:]), dataset.words.apply(lambda x: "risk" in x[-5:])
        prj, risk=self.pre_suffix_strip("prj", dataset[prj_mask]), self.pre_suffix_strip("risk", dataset[risk_mask])
        return prj, risk

    def stopword_detection(self, trainset: pd.DataFrame = None):
        """extract out the keyword only in nrisk"""
        if not trainset: trainset=self.df
        trainset.reset_index(drop=True, inplace=True)
        trainset.words = trainset.words.str.lower()

        prj, risk=self.keyword_identify(trainset)
        stk=trainset[~trainset.index.isin(prj.index) & ~trainset.index.isin(risk.index)]
        stk["word"]=stk.words
        prj, risk=self.nltk_tagger(prj), self.nltk_tagger(risk)
        stk=self.nltk_tagger(stk)
        # stk=self.stakeholder_filter(pd.DataFrame(stk, columns=["words"]))

        self.update_stop_remain([*prj, *risk])

    def nltk_tagger(self, trainset: pd.DataFrame = None):
        """use NLTK tag on words, then only get Noun-realted words
        store the rest corpous as stop_words, named NLTK_reuslt.xlsx"""
        trainset["word type"] = trainset.word.apply(lambda row: nltk.pos_tag(nltk.word_tokenize(row))[0][1])
        nnres = trainset[trainset["word type"].apply(lambda x: x in ["NN", "NNP", "NNS", "NNPS"])]
        remain_res = trainset.drop(nnres.index)
        non_nnstopwords = remain_res.words.to_list()
        return non_nnstopwords

    def LLM_recoginize(self, require):
        key = "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"
        llm = OpenAI(temperature=0, openai_api_key=key)
        text = f"please split sentence by '|' as a unit and filter out any word that cannot be an organization or " \
               f"stakeholder, return me a list of combination, given the text '{require}' "
        response = llm(text)
        filter_out = [val.lower() for val in response.strip("\n").split(", ")]
        return filter_out

    def multi_sheet_excel_writer(self, dataset: pd.DataFrame, file_name):
        writer = pd.ExcelWriter(f"{file_name}.xlsx")
        for name, val in dataset.groupby("word type"):
            val.to_excel(writer, index=False)

    def risk_and_project(self, data):
        """kick out project and risk stock phrase within dataset"""
        if "project" in data or "risk" in data: return False
        return True

    def stopword_keep(self, stp: list):
        remain = list()
        for val in stp:
            if len(val) > 25:
                remain.append(val)
                stp.remove(val)
        return stp, remain

    def stakeholder_filter(self, nnres):
        """by counting the frequency, extract first 2K words and combine together"""
        filter_stk = nnres[nnres.words.apply(self.risk_and_project)]
        lenfile=len(filter_stk)
        require1, require2 = "|".join(filter_stk.words[0:int(lenfile*0.3)].to_list()), "|".join(
            filter_stk.words[int(lenfile*0.3):int(lenfile*0.5)].to_list())
        filter_out = [*self.LLM_recoginize(require1), *self.LLM_recoginize(require2)]
        """now we get the meanness stakeholder word, which stored in "filter_out", we split them from dataset"""
        mask = self.stk_key.words.str.contains("|".join(filter_out))
        self.stk_key = self.stk_key[~mask]
        return filter_out

    def update_stopword(self, stopword):
        with open(self.stop_words, "w") as file:
            for val in stopword: file.write(val + "\n")

    def update_remain(self, remain):
        with open(self.unknow_long_phrase, "w") as file:
            for val in remain: file.write(val + "\n")

    def update_stop_remain(self, filter_out):
        with open(self.stop_words, "r") as file:
            stopword = [line.strip() for line in file.readlines()]
        filter_out, remain = self.stopword_keep(filter_out)
        stopword = list({*stopword, *filter_out})
        self.update_stopword(stopword)
        self.update_remain(remain)


if __name__ == "__main__":
    df_path=pathprefix+"\\FP_growth_result\\to_one.csv"
    dataframe=pd.read_csv(df_path, sep=",")
    freq=dataframe.to.value_counts().reset_index(name="count")
    freq["words"]=freq["index"]
    freq=freq.drop(["index"], axis=1)

    initialize = Filtering(freq)
    initialize.stopword_detection()