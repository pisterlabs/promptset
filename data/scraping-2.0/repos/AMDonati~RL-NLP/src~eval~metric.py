import heapq
import logging
import operator
import os

os.environ['TRANSFORMERS_CACHE'] = "/cache"

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from nlgeval.pycocoevalcap.cider.cider import Cider
from nltk.translate.meteor_score import meteor_score
# import nltk
from tools.refer.evaluation.tokenizer.ptbtokenizer import PTBTokenizer
from torch.nn.utils.rnn import pad_sequence
from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
from transformers import AutoModelWithLMHead, AutoTokenizer

from RL_toolbox.reward import rewards
# If modifying these scopes, delete the file token.pickle.
from data_provider.CLEVR_Dataset import CLEVR_Dataset
from models.language_model import ClevrLanguageModel
from RL_toolbox.reward import get_vocab
import nltk
from RL_toolbox.globals import vilbert_model, gpt2_model, gpt2_tokenizer, compute_score_with_logits

# nltk.download('wordnet')
try:
    from vr.utils import load_execution_engine, load_program_generator
except ImportError:
    print("VR NOT IMPORTED!!")

SCOPES = ['https://www.googleapis.com/auth/drive.metadata.readonly']

logger = logging.getLogger()


def _strip(s):
    return s.strip()


class Metric:
    def __init__(self, agent, train_test, key, type, env, trunc, sampling):
        self.measure = []
        self.metric = []
        self.metric_history = []
        self.metric_diversity = []
        self.metric_diversity_history = []
        self.idxs_to_select = []
        self.idx_to_select = None
        self.idx_step = 0
        self.idx_word = 0
        self.idx_compute = 0
        self.idx_write = 1
        self.dataset = env.dataset
        self.out_path = agent.out_path
        self.writer = agent.writer
        self.language_model = agent.truncation.language_model
        self.policy = agent.policy
        self.reward_type = env.reward_type
        self.type = type
        self.key = key
        self.train_test = train_test
        self.id = "_".join([env.mode, trunc, sampling])
        self.env_mode = env.mode
        self.env = env
        self.trunc = trunc
        self.sampling = sampling
        # self.dict_metric, self.dict_stats = {}, {}  # for csv writing.
        self.name = train_test + "_" + self.id + '_' + self.key
        self.out_csv_file = os.path.join(self.out_path, "metrics", self.name + ".csv")
        self.out_div_csv_file = os.path.join(self.out_path, "diversity", self.name + ".csv")
        self.stats = {}
        self.stats_div = {}
        self.to_tensorboard = True if key in metrics_to_tensorboard else False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fill(self, **kwargs):
        self.fill_(**kwargs)
        self.idx_word += 1
        self.idx_step += 1

    def compute(self, **kwargs):
        self.compute_(**kwargs)
        self.measure = []
        self.idx_word = 0
        self.idx_step = 0
        self.idx_compute += 1

    def reset(self):
        self.idx_word = 0

    def write(self, idx_to_select=False):
        if self.to_tensorboard:
            if self.type == "scalar":
                self.writer.add_scalar(self.name, np.mean(self.metric), self.idx_write)
            elif self.type == "text":
                self.writer.add_text(self.name, '  \n'.join(self.metric[-1:]), self.idx_write)
        self.idx_write += 1
        self.metric_history.extend(self.metric)
        if self.sampling != "greedy":
            self.metric_diversity.extend(self.metric)
        self.metric = []
        if idx_to_select and self.sampling == "sampling_ranking_lm":
            self.idx_to_select = self.idx_compute - 1

    def write_div(self, **kwargs):
        if self.type == "scalar" and self.metric_diversity:
            metric_diversity = [np.mean(self.metric_diversity), np.std(self.metric_diversity),
                                np.max(self.metric_diversity), np.min(self.metric_diversity)]
            self.metric_diversity_history.append(metric_diversity)
            self.metric_diversity = []

    def log(self, **kwargs):
        pass

    def get_stats(self, serie, num_episodes, idx_to_keep=None, num_diversity=10):
        if idx_to_keep is not None and serie.size == num_diversity * num_episodes:
            serie_reshaped = serie.to_numpy().reshape(-1, num_diversity)
            values = np.take_along_axis(serie_reshaped, np.array(idx_to_keep).reshape(-1, 1), 1)
            serie_to_keep = pd.Series(values.ravel())
        else:
            serie_to_keep = serie
        return [serie_to_keep.mean(), serie_to_keep.std(), serie_to_keep.size]

    def get_stats_div(self, df):
        return df.mean().to_dict()

    def post_treatment_(self):
        pass

    def filter_reranking(self, num_episodes, idxs_to_select):
        if idxs_to_select is not None and self.sampling == "sampling_ranking_lm" and len(
                self.metric_history) == num_episodes * 10:
            self.metric_history = np.array(self.metric_history)
            self.metric_history = list(self.metric_history[idxs_to_select])

    def save_series_and_stats(self, idx_to_keep, num_episodes):
        serie = pd.Series(self.metric_history)
        serie.to_csv(self.out_csv_file, index=False, header=False)
        if self.type == "scalar":
            self.stats = {self.key: self.get_stats(serie, idx_to_keep=idx_to_keep, num_episodes=num_episodes)}
            if self.metric_diversity_history:
                df = pd.DataFrame(data=self.metric_diversity_history, columns=["mean", "std", "max", "min"])
                df.to_csv(self.out_div_csv_file)
                self.stats_div = {self.key: self.get_stats_div(df)}

    def post_treatment(self, num_episodes, idx_to_keep=None):
        self.post_treatment_()
        self.save_series_and_stats(idx_to_keep, num_episodes)

        # ----------------------------------  TRAIN METRICS -------------------------------------------------------------------------------------


class VAMetric(Metric):
    '''Display the valid action space in the training log.'''

    def __init__(self, agent, train_test, env, trunc, sampling):
        Metric.__init__(self, agent, train_test, "valid_actions", "text", env, trunc, sampling)

    def fill_(self, **kwargs):
        state_decoded = self.dataset.question_tokenizer.decode(kwargs["state"].text.numpy()[0],
                                                               ignored=['<PAD>'])
        if self.language_model.init_text is not None:
            state_decoded = self.language_model.init_text_short + "\n" + state_decoded
        string = ""
        if kwargs["valid_actions"] is not None:
            top_words_decoded = [self.dataset.question_tokenizer.decode([va]) for va in
                                 kwargs["valid_actions"].cpu().numpy()[0]]
            weights_words = ["{}/{:.3f}".format(word, weight, number=3) for word, weight in
                             zip(top_words_decoded, kwargs["actions_probs"].cpu().detach().numpy()[0])]
            string = "next possible words for {} : {}".format(state_decoded, ", ".join(weights_words))
        self.measure.append(string)

    def compute_(self, **kwargs):
        self.metric = self.measure

    def log(self, **kwargs):
        if kwargs["valid_actions"] is not None:
            logger.info('---------------------Valid action space------------------------------')
            logger.info('\n'.join(self.metric))
            logger.info('---------------------------------------------------------------------')

    def write(self):
        pass


class SizeVAMetric(Metric):
    '''Compute the average size of the truncated action space during training for truncation functions proba_thr & sample_va'''

    def __init__(self, agent, train_test, env, trunc, sampling):
        Metric.__init__(self, agent, train_test, "size_valid_actions", "scalar", env, trunc, sampling)
        self.counter = 0

    def fill_(self, **kwargs):
        if kwargs["valid_actions"] is not None:
            self.measure.append(kwargs["valid_actions"].size(1))

    def compute_(self, **kwargs):
        self.metric.append(np.mean(self.measure))


class SumProbsOverTruncated(Metric):
    '''Compute the sum of the probabilities the action space given by the language model.'''

    def __init__(self, agent, train_test, env, trunc, sampling):
        Metric.__init__(self, agent, train_test, "sum_probs_truncated", "scalar", env, trunc, sampling)

    def fill_(self, **kwargs):
        sum_over_truncated_space = 1
        if kwargs["valid_actions"] is not None:
            sum_over_truncated_space = torch.gather(kwargs["dist"].probs, -1,
                                                    kwargs["valid_actions"]).sum().cpu().detach().numpy()
        self.measure.append(float(sum_over_truncated_space))

    def compute_(self, **kwargs):
        self.metric.append(np.mean(self.measure))


# --------------------  TEST METRICS ----------------------------------------------------------------------------------------------------------------------------
class DialogMetric(Metric):
    """Display the test dialog."""

    def __init__(self, agent, train_test, env, trunc, sampling):
        Metric.__init__(self, agent, train_test, "dialog", "text", env, trunc, sampling)
        # self.out_dialog_file = os.path.join(self.out_path, self.train_test + '_' + self.key + '.txt')
        # self.h5_dialog_file = os.path.join(self.out_path, self.train_test + '_' + self.key + '.h5')
        self.generated_dialog = {}

    def fill_(self, **kwargs):
        pass

    def compute_(self, **kwargs):
        with torch.no_grad():
            state_decoded = self.dataset.question_tokenizer.decode(kwargs["state"].text[:, :].numpy()[0],
                                                                   ignored=[])
            if self.reward_type == 'vqa' or self.reward_type == "vilbert" or self.reward_type == "vilbert_rank2":
                pred_answer = [int(kwargs["pred_answer"].squeeze().detach().numpy())]
                pred_answer_decoded = self.dataset.answer_tokenizer.decode(text=pred_answer)
                ref_answer_decoded = self.dataset.answer_tokenizer.decode(kwargs["ref_answer"].view(1).numpy())
                ref_question_decoded = kwargs["ref_questions_decoded"]
                string = ' IMG {} - question index {}:'.format(kwargs["img_idx"],
                                                               kwargs[
                                                                   "question_idx"]) + '\n' + 'DIALOG:' + state_decoded + '\n' + 'REF QUESTION:' + \
                         ref_question_decoded[
                             0] + '\n' + 'VQA ANSWER:' + pred_answer_decoded + '\n' + 'TRUE ANSWER:' + ref_answer_decoded + '\n' + '-' * 40
            else:
                closest_question_decoded = kwargs["closest_question"]
                string = 'IMG {}:'.format(kwargs[
                                              "img_idx"]) + state_decoded + '\n' + 'CLOSEST QUESTION:' + closest_question_decoded + '\n' + '-' * 40

        self.metric.append(string)

    def log(self, **kwargs):
        if len(self.metric) > 0:
            logger.info(self.metric[-1])
            logger.info("-" * 40)

    def write_to_csv(self):
        '''save padded array of generated dialog for later use (for example with word cloud)'''
        if self.train_test != "train":
            for key, dialog in self.generated_dialog.items():
                generated_dialog = pad_sequence(dialog, batch_first=True).cpu().numpy()
                with h5py.File(self.h5_dialog_file, 'w') as f:
                    f.create_dataset(key, data=generated_dialog)


class DialogImageMetric(Metric):
    '''Display the Dialog on a html format at test time.'''

    def __init__(self, agent, train_test, env, trunc, sampling):
        Metric.__init__(self, agent, train_test, "dialog_image", "text", env, trunc, sampling)
        self.generated_dialog = []
        self.condition_answer = agent.policy.condition_answer
        self.mode = agent.env.mode
        image_id_file = "clevr" if self.dataset.__class__ == CLEVR_Dataset else "coco"
        image_id_file = os.path.join("data", "drive", image_id_file + ".csv")
        self.list_image_ids = pd.read_csv(image_id_file, index_col="id_image")
        self.out_html_file = os.path.join(self.out_path, "metrics", self.name + ".html")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fill_(self, **kwargs):
        if kwargs["valid_actions"] is not None:
            true_action = kwargs["ref_question"].view(-1)[kwargs["timestep"]]
            in_va = true_action.cpu() in kwargs["valid_actions"].cpu()
            _, indices = torch.sort(kwargs["log_probas_lm"], descending=True)
            rank = int(torch.nonzero(indices.squeeze().cpu() == true_action).squeeze().numpy())

            self.measure.append([in_va, rank])

    def compute_(self, **kwargs):
        with torch.no_grad():
            state_decoded = self.dataset.question_tokenizer.decode(text=kwargs["state"].text[:, 1:].numpy()[0],
                                                                   ignored=[])
            values = {}
            values["img"] = kwargs["img_idx"]
            values["question"] = state_decoded
            values["reward"] = round(kwargs["reward"], 3)
            values["ref_questions"] = kwargs["ref_questions_decoded"]
            values["in_valid_actions"] = self.measure

            if self.condition_answer != "none":
                ref_answer_decoded = self.dataset.answer_tokenizer.decode(kwargs["ref_answer"].view(1).numpy())
                values["ref_answer"] = ref_answer_decoded

            if kwargs["pred_answer"] != None:
                pred_answer = [int(kwargs["pred_answer"].squeeze().detach().numpy())]
                pred_answer_decoded = self.dataset.answer_tokenizer.decode(text=pred_answer)
                values["pred_answer"] = pred_answer_decoded

            dialog = ["{} : {}".format(key, value) for key, value in values.items()]
            dialog.append("-" * 70)
            dialog = " \n ".join(dialog)
            self.metric.append(dialog)

            id = self.get_id_image(kwargs["img_idx"])
            # if id is not None:
            url = "https://drive.google.com/uc?export=view&id={}".format(id)
            values["link"] = "<img src={}>".format(url)
            values["closest_question"] = kwargs["closest_question"]
            self.generated_dialog.append(values)

    def get_id_image(self, id_image):

        try:
            if self.dataset.__class__ == CLEVR_Dataset:
                mode = 0 if self.mode == "train" else 1
                id_drive = self.list_image_ids.loc[id_image]["id_google"].iloc[mode]
            else:
                id_drive = self.list_image_ids.loc[id_image]["id_google"]

        except:
            id_drive = None
        finally:
            return id_drive

    def post_treatment_(self):
        num_last_episodes = min(50, len(self.generated_dialog))
        to_html = lambda x: "".join(["<li>{} : {}</li>".format(key, value) for key, value in x.items()])
        html_str = ["<tr><ul>" + to_html(x) + "</ul></tr>" for x in self.generated_dialog[-num_last_episodes:]]
        html_str = "".join(html_str)
        html_str = "<table>" + html_str + "</table>"
        f = open(self.out_html_file, "x")
        f.write(html_str)
        f.close()


class PPLDialogfromLM(Metric):
    '''Computes the PPL of the Language Model over the generated dialog'''

    def __init__(self, agent, train_test, env, trunc, sampling):
        Metric.__init__(self, agent, train_test, "ppl_dialog_lm", "scalar", env, trunc, sampling)
        self.min_data = agent.env.min_data
        self.device = agent.device
        self.get_lm_model(agent)

    def get_lm_model(self, agent):
        if self.dataset.__class__ == CLEVR_Dataset:
            lm_model = torch.load("output/lm_model/model.pt", map_location=torch.device('cpu'))
        else:
            if self.min_data:
                lm_model = torch.load("output/vqa_lm_model_smallvocab/model.pt", map_location=torch.device('cpu'))
            else:
                lm_model = torch.load("output/vqa_lm_model/model.pt", map_location=torch.device('cpu'))
        lm_model.eval()
        self.pretrained_lm = ClevrLanguageModel(pretrained_lm=lm_model, dataset=self.dataset,
                                                tokenizer=self.dataset.question_tokenizer, device=agent.device)

    def fill_(self, **kwargs):
        pass

    def get_ppl(self, inputs):
        with torch.no_grad():
            loss = torch.nn.CrossEntropyLoss(ignore_index=0)
            log_probas, logits = self.pretrained_lm.language_model(inputs.to(self.device))
            shift_logits = logits[..., :-1, :].contiguous().to(self.device)
            shift_logits[:, self.pretrained_lm.unk_idx] = shift_logits.min().to(self.device)
            shift_labels = inputs[..., 1:].contiguous().to(self.device)
            loss_ = loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            ppls = torch.exp(loss_)
            return ppls.view(-1).tolist()

    def compute_(self, **kwargs):
        ppls = self.get_ppl(inputs=kwargs["state"].text.clone())
        self.metric.extend(ppls)

    def get_min_ppl_idxs(self, num_diversity):
        ppls = torch.tensor(self.metric_history).view(-1, num_diversity)
        idx_to_keep = torch.argmin(ppls, dim=1).tolist()
        return idx_to_keep


class PPLDialogfromLMExt(PPLDialogfromLM):
    def __init__(self, agent, train_test, env, trunc, sampling):
        Metric.__init__(self, agent, train_test, "ppl_dialog_lm_ext", "scalar", env, trunc, sampling)
        self.min_data = agent.env.min_data
        self.device = agent.device
        self.lm_path = "output/lm_ext/model.pt"
        self.lm_model = torch.load(self.lm_path, map_location=self.device)
        self.pretrained_lm = ClevrLanguageModel(pretrained_lm=self.lm_model, dataset=self.dataset,
                                                tokenizer=self.dataset.question_tokenizer, device=self.device)


class LanguageScore(Metric):
    '''Compute the perplexity of a pretrained language model (GPT) on the generated dialog.'''

    def __init__(self, agent, train_test, env, trunc, sampling):
        Metric.__init__(self, agent, train_test, "language_score", "scalar", env, trunc, sampling)
        self.lm_model = gpt2_model
        self.tokenizer = gpt2_tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.questions = []
        self.batch_size = 1000

    def fill_(self, **kwargs):
        pass

    def process_batch(self, questions):
        loss = torch.nn.CrossEntropyLoss(reduction="none")
        inputs = self.tokenizer(questions, padding=True, truncation=True, return_tensors="pt")
        labels = inputs["input_ids"].clone()
        labels[inputs["attention_mask"] == 0] = -100
        outputs = self.lm_model(**inputs, labels=labels)
        shift_logits = outputs["logits"][..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_ = loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(labels.size(0),
                                                                                               labels.size(1) - 1)
        masked_loss = loss_.sum(dim=-1)
        masked_loss /= inputs["attention_mask"].sum(dim=-1)
        ppls_per_sentence = torch.exp(masked_loss)
        return ppls_per_sentence.view(-1).tolist()

    def reset(self):
        self.questions = []

    def compute_(self, **kwargs):
        if kwargs["state"].text.shape[-1] > 1:
            state_decoded = self.dataset.question_tokenizer.decode(kwargs["state"].text[:, 1:].cpu().numpy()[0])
            self.questions.append(state_decoded)
        if len(self.questions) == self.batch_size:
            ppl = self.process_batch(self.questions)
            self.metric.extend(ppl)
            self.reset()

    def get_min_ppl_idxs(self, num_diversity):
        if len(self.questions) > 0:
            ppl = self.process_batch(self.questions)
            self.metric_history.extend(ppl)
            self.reset()
        ppls = torch.tensor(self.metric_history).view(-1, num_diversity)
        idx_to_keep = torch.argmin(ppls, dim=1).tolist()
        return idx_to_keep

    def post_treatment(self, num_episodes, idx_to_keep=None):
        if len(self.questions) > 0:
            ppl = self.process_batch(self.questions)
            self.metric_history.extend(ppl)
            self.reset()
        # self.filter_reranking(num_episodes, idx_to_keep)
        self.post_treatment_()
        self.save_series_and_stats(idx_to_keep, num_episodes)


class Return(Metric):
    def __init__(self, agent, train_test, env, trunc, sampling):
        Metric.__init__(self, agent, train_test, "return", "scalar", env, trunc, sampling)

    def fill_(self, **kwargs):
        self.measure.append(kwargs["reward"])

    def compute_(self, **kwargs):
        ep_return = np.sum(self.measure)
        self.metric.append(ep_return)


class BleuMetric(Metric):
    '''Compute the bleu score over the ref questions and the generated dialog.'''

    def __init__(self, agent, train_test, env, trunc, sampling):
        Metric.__init__(self, agent, train_test, "bleu", "scalar", env, trunc, sampling)
        if "bleu" in agent.env.reward_type:
            self.function = agent.env.reward_func
        else:
            self.function = rewards["bleu_sf2"]()

    def fill_(self, **kwargs):
        if kwargs["done"]:
            question_decoded = self.dataset.question_tokenizer.decode(kwargs["new_state"].text.numpy()[0],
                                                                      ignored=["<SOS>"],
                                                                      stop_at_end=True)
            ref_questions = kwargs["ref_questions_decoded"]
            score, _, _ = self.function.get(ep_questions_decoded=ref_questions, question=question_decoded, done=True)
            self.measure.append(score)

    def compute_(self, **kwargs):
        self.metric.append(np.mean(self.measure))


class SelfBleuImageMetric(Metric):
    '''Compute the self bleu score on generated sentences for one image'''

    def __init__(self, agent, train_test, env, trunc, sampling):
        Metric.__init__(self, agent, train_test, "selfbleu", "scalar", env, trunc, sampling)
        if "bleu" in agent.env.reward_type:
            self.function = agent.env.reward_func
        else:
            self.function = rewards["bleu_sf2"]()
        self.questions = []
        self.out_questions_csv_file = os.path.join(self.out_path, "metrics", self.name + "_questions.csv")

    def fill_(self, **kwargs):
        pass

    def compute_(self, **kwargs):
        if self.sampling != "greedy":
            question_decoded = self.dataset.question_tokenizer.decode(kwargs["state"].text.numpy()[0],
                                                                      ignored=["<SOS>"],
                                                                      stop_at_end=True)
            self.questions.append(question_decoded)
            if (kwargs["idx_diversity"] == kwargs["num_diversity"] - 1) and kwargs["num_diversity"] > 1:
                scores = []
                for i, question in enumerate(self.questions):
                    ref_questions = np.delete(self.questions, i)
                    score, _, _ = self.function.get(ep_questions_decoded=ref_questions, question=question, done=True)
                    scores.append(score)
                self.metric.append(np.mean(scores))
                self.questions = []


class HistogramOracle(Metric):
    """Compute the Histogram of Correct Answers."""

    def __init__(self, agent, train_test, env, trunc, sampling):
        Metric.__init__(self, agent, train_test, "histogram_answers", "text", env, trunc, sampling)
        self.metric_history = {}
        self.answer_history = {}
        self.out_png_file = os.path.join(self.out_path, "metrics", self.name + ".png")

    def fill_history(self, history, ref_answer_decoded, reward):
        if ref_answer_decoded in history.keys():
            history[ref_answer_decoded] += reward
        else:
            history[ref_answer_decoded] = reward
        return history

    def fill_(self, **kwargs):
        if self.reward_type == "vilbert" or self.reward_type == "vqa":
            if kwargs["done"]:
                ref_answer_decoded = self.dataset.answer_tokenizer.decode(kwargs["ref_answer"].view(1).numpy())
                if kwargs["reward"] == 1.:
                    self.metric_history = self.fill_history(self.metric_history, ref_answer_decoded, kwargs["reward"])
                    self.answer_history = self.fill_history(self.answer_history, ref_answer_decoded, 1)
                else:
                    self.answer_history = self.fill_history(self.answer_history, ref_answer_decoded, 1)

    def compute_(self, **kwargs):
        pass

    def write(self, **kwargs):
        pass

    def get_top_k_values(self, k=25):
        if len(self.metric_history) > k:
            k_keys_sorted_by_values = heapq.nlargest(k, self.metric_history, key=self.metric_history.get)
            top_k_dict = {k: v for k, v in self.metric_history.items() if k in k_keys_sorted_by_values}
        else:
            top_k_dict = self.metric_history
        return top_k_dict

    def post_treatment(self, num_episodes, idx_to_keep=None):
        if self.reward_type == "vilbert" or self.reward_type == "vqa":
            self.post_treatment_()
            metric_history_sorted = dict(sorted(self.metric_history.items(), key=operator.itemgetter(1), reverse=True))
            answer_history_sorted = {k: self.answer_history[k] for k in list(metric_history_sorted.keys())}
            df = pd.DataFrame.from_dict([metric_history_sorted, answer_history_sorted])
            df = df.transpose()
            df.to_csv(self.out_csv_file, index=True, header=["reward = 1", "answer freq"])

    def post_treatment_(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(30, 15))
        top_k_metric_history = self.get_top_k_values()
        top_k_answer_history = {k: self.answer_history[k] for k in list(top_k_metric_history.keys())}
        ax1.bar(list(top_k_metric_history.keys()), top_k_metric_history.values())
        ax2.bar(list(top_k_metric_history.keys()), top_k_answer_history.values())
        ax1.tick_params(labelsize=18)
        ax2.tick_params(labelsize=18)
        plt.savefig(self.out_png_file)


class UniqueWordsMetric(Metric):
    '''Compute the ratio of Unique Words for the set of questions generated for each image. Allows to measure vocabulary diversity.'''

    def __init__(self, agent, train_test, env, trunc, sampling):
        Metric.__init__(self, agent, train_test, "ttr", "scalar", env, trunc, sampling)
        self.measure_history = []
        self.threshold = 10

    def fill_(self, **kwargs):
        if kwargs["done"]:
            self.measure_history.append(list(kwargs["new_state"].text.squeeze().cpu().numpy()[1:]))

    def compute_(self, **kwargs):
        if self.idx_compute > self.threshold and "sampling" in self.id:
            arr = [item for sublist in self.measure_history[-self.threshold:] for item in sublist]
            unique_tokens = np.unique(arr)
            diversity_metric = len(unique_tokens) / len(arr) if len(arr) > 0 else 0
            self.metric.append(diversity_metric)


class VilbertRecallMetric(Metric):
    '''Compute the oracle score over the ref answer and the generated dialog.'''

    def __init__(self, agent, train_test, env, trunc, sampling):
        Metric.__init__(self, agent, train_test, "vilbert_recall", "scalar", env, trunc, sampling)

        self.ranks = []
        self.rewards = []
        if "vilbert" in env.reward_type:
            self.model = env.reward_func.model
        else:
            self.model = vilbert_model
        self.batch_size = 128
        self.reset()
        if self.env.reward_type in ["vilbert_recall"]:
            self.function = agent.reward_func
        else:
            self.function = rewards["vilbert_recall"](path="output/vilbert_vqav2/model.bin",
                                                      vocab="output/vilbert_vqav2/bert_base_6layer_6conect.json",
                                                      env=self.env)

    def fill_(self, **kwargs):
        pass

    def reset(self):
        self.features = []
        self.spatials = []
        self.encoded_question = []
        self.segment_ids = []
        self.input_mask = []
        self.image_mask = []
        self.co_attention_mask = []
        self.targets = []
        self.question_id = []
        self.task_tokens = []
        self.ep_questions_decoded = []

    def process_batch(self):

        vil_prediction, vil_prediction_gqa, vil_logit, vil_binary_prediction, vil_tri_prediction, vision_prediction, vision_logit, linguisic_prediction, linguisic_logit, _ = self.model(
            torch.cat(self.encoded_question, dim=0),
            torch.cat(self.features, dim=0),
            torch.cat(self.spatials, dim=0),
            torch.cat(self.segment_ids, dim=0),
            torch.cat(self.input_mask, dim=0),
            torch.cat(self.image_mask, dim=0),
            torch.cat(self.co_attention_mask, dim=0),
            torch.cat(self.task_tokens, dim=0),
        )

        if self.env.reduced_answers:
            mask = torch.ones_like(vil_prediction) * float("-Inf")
            mask[:, self.dataset.reduced_answers.squeeze().long()] = vil_prediction[:,
                                                                     self.dataset.reduced_answers.squeeze().long()]
            vil_prediction = mask
        sorted_logits, sorted_indices = torch.sort(vil_prediction, descending=True)
        targets = torch.cat(self.targets, dim=0)
        masked_preds = (-vil_prediction * (targets != 0).int().float())
        masked_preds[masked_preds == 0] = -float("Inf")
        maxs_of_targets, argmaxs_of_targets = masked_preds.max(dim=1)
        ranks = (argmaxs_of_targets.unsqueeze(dim=1) == sorted_indices).nonzero()[:, 1]
        reward = compute_score_with_logits(vil_prediction, targets, device=self.device)
        self.ranks.extend(ranks.tolist())
        self.rewards.extend(reward.sum(dim=1).tolist())

    def compute_(self, **kwargs):

        (features,
         spatials,
         image_mask,
         real_question,
         target,
         real_input_mask,
         real_segment_ids,
         co_attention_mask,
         question_id) = self.dataset.get_data_for_ViLBERT(self.env.env_idx)

        question_tokens = kwargs["state"].text.numpy().ravel()
        question = self.dataset.question_tokenizer.decode(question_tokens)  #
        encoded_question = self.dataset.reward_tokenizer.encode(question)
        encoded_question = self.dataset.reward_tokenizer.add_special_tokens_single_sentence(encoded_question)

        segment_ids, input_mask, encoded_question = self.dataset.get_masks_for_tokens(encoded_question)
        segment_ids, input_mask, encoded_question = torch.tensor(segment_ids), torch.tensor(input_mask), torch.tensor(
            encoded_question).view(-1)
        task_tokens = encoded_question.new().resize_(encoded_question.size(0), 1).fill_(1)

        self.features.append(features.unsqueeze(dim=0))
        self.spatials.append(spatials.unsqueeze(dim=0))
        self.encoded_question.append(encoded_question.unsqueeze(dim=0))
        self.segment_ids.append(segment_ids.unsqueeze(dim=0))
        self.input_mask.append(input_mask.unsqueeze(dim=0))
        self.image_mask.append(image_mask.unsqueeze(dim=0))
        self.co_attention_mask.append(co_attention_mask.unsqueeze(dim=0))
        self.targets.append(target.unsqueeze(dim=0))
        self.question_id.append(question_id)
        self.task_tokens.append(task_tokens.unsqueeze(dim=0))
        self.ep_questions_decoded.append(kwargs["ref_questions_decoded"])

        if len(self.features) == self.batch_size:
            self.process_batch()
            self.reset()

    def filter_reranking(self, num_episodes, idxs_to_select):
        if idxs_to_select is not None and self.sampling == "sampling_ranking_lm" and len(
                self.metric_history) == num_episodes * 10:
            self.ranks = np.array(self.ranks)
            self.ranks = list(self.ranks[idxs_to_select])
            self.rewards = np.array(self.rewards)
            self.rewards = list(self.rewards[idxs_to_select])

    def post_treatment(self, num_episodes, idx_to_keep=None):
        if len(self.features) > 0:
            self.process_batch()
            self.reset()
        # self.filter_reranking(num_episodes, idx_to_keep)

        serie_ranks = pd.Series(self.ranks)
        serie_rewards = pd.Series(self.rewards)
        serie_recall_5 = (serie_ranks < 5).astype(int)

        ranks_out_file = os.path.join(self.out_path, "metrics", self.name + "_ranks.csv")
        serie_ranks.to_csv(ranks_out_file, index=False, header=False)
        if self.type == "scalar":
            self.stats = {"ranks": self.get_stats(serie_ranks, idx_to_keep=idx_to_keep, num_episodes=num_episodes),
                          "recall_5": self.get_stats(serie_recall_5, idx_to_keep=idx_to_keep,
                                                     num_episodes=num_episodes),
                          "oracle_score": self.get_stats(serie_rewards, idx_to_keep=idx_to_keep,
                                                         num_episodes=num_episodes)}

        ranks_out_file = os.path.join(self.out_path, "metrics", self.name + "_ranks.csv")
        rewards_out_file = os.path.join(self.out_path, "metrics", self.name + "_rewards.csv")

        serie_ranks.to_csv(ranks_out_file, index=False, header=False)
        serie_rewards.to_csv(rewards_out_file, index=False, header=False)


class OracleClevr(Metric):
    def __init__(self, agent, train_test, env, trunc, sampling):
        Metric.__init__(self, agent, train_test, "oracle_recall", "scalar", env, trunc, sampling)
        vocab = "data/closure_vocab.json"
        path = "output/vqa_model_film/model.pt"
        self.execution_engine, ee_kwargs = load_execution_engine(path)
        self.execution_engine.to(self.device)
        self.execution_engine.eval()
        self.program_generator, pg_kwargs = load_program_generator(path)
        self.program_generator.to(self.device)
        self.program_generator.eval()
        self.vocab = vocab
        self.vocab_questions_vqa = get_vocab('question_token_to_idx', self.vocab)
        # self.vocab_questions_vqa.update({"<pad>": 0, "<sos>": 1, "<eos>": 2})
        self.trad_dict = {value: self.vocab_questions_vqa[key.lower()] for key, value in
                          self.dataset.vocab_questions.items() if
                          key.lower() in self.vocab_questions_vqa}
        self.decoder_dict = {value: key for key, value in self.vocab_questions_vqa.items()}

        self.reset()
        self.batch_size = 30

    def trad(self, input):
        idx_vqa = [self.trad_dict[idx] for idx in input if idx in self.trad_dict]
        idx_vqa.insert(0, 1)  # add SOS token.
        idx_vqa.append(2)  # add EOS token.
        return torch.tensor(idx_vqa).unsqueeze(dim=0).squeeze().to(self.device)

    def decode(self, input):
        return " ".join([self.decoder_dict[word] for word in input])

    def fill_(self, **kwargs):
        pass

    def process_batch(self):
        programs_pred = self.program_generator(pad_sequence(self.questions, batch_first=True))
        scores = self.execution_engine(torch.cat(self.imgs), programs_pred)
        sorted = torch.argsort(scores, descending=True)
        ranks = torch.nonzero(sorted.cpu() == torch.cat(self.answers).view((sorted.size(0), -1)))
        self.metric_history.extend(ranks[:, 1].tolist())

    def reset(self):
        self.questions = []
        self.imgs = []
        self.answers = []

    def compute_(self, **kwargs):
        self.questions.append(self.trad(kwargs["state"].text.squeeze().cpu().numpy()))
        self.imgs.append(kwargs["state"].img.to(self.device))
        self.answers.append(kwargs["ref_answer"].view(-1))

        if len(self.questions) == self.batch_size:
            self.process_batch()
            self.reset()

    def post_treatment(self, num_episodes, idx_to_keep=None):
        if len(self.questions) != 0:
            self.process_batch()
            self.reset()
        # self.filter_reranking(num_episodes, idx_to_keep)
        serie_ranks = pd.Series(self.metric_history)
        serie_recall_5 = (serie_ranks < 5).astype(int)
        serie_score = (serie_ranks == 0).astype(int)

        ranks_out_file = os.path.join(self.out_path, "metrics", self.name + "_ranks.csv")
        serie_ranks.to_csv(ranks_out_file, index=False, header=False)
        if self.type == "scalar":
            self.stats = {
                "oracle_score": self.get_stats(serie_score, idx_to_keep=idx_to_keep, num_episodes=num_episodes),
                "recall_5": self.get_stats(serie_recall_5, idx_to_keep=idx_to_keep, num_episodes=num_episodes)}


class CiderMetric(Metric):
    def __init__(self, agent, train_test, env, trunc, sampling):
        Metric.__init__(self, agent, train_test, "cider", "scalar", env, trunc, sampling)
        self.score_function = Cider()
        self.tokenizer = PTBTokenizer()
        self.candidates = []
        self.refs = []

    def fill_(self, **kwargs):
        pass

    def compute_(self, **kwargs):
        question_decoded = self.dataset.question_tokenizer.decode(kwargs["state"].text.numpy()[0],
                                                                  ignored=["<SOS>"], stop_at_end=True)
        ref_questions = kwargs["ref_questions_decoded"][0]
        self.candidates.append(question_decoded)
        self.refs.append([ref_questions])

    def post_treatment_(self):
        refs = {idx: list(map(_strip, ref)) for (idx, ref) in enumerate(self.refs)}
        hyps = {idx: [lines.strip()] for (idx, lines) in enumerate(self.candidates)}
        score, scores = self.score_function.compute_score(refs, hyps)
        self.metric_history.extend(scores)


class MeteorMetric(Metric):
    def __init__(self, agent, train_test, env, trunc, sampling):
        Metric.__init__(self, agent, train_test, "meteor", "scalar", env, trunc, sampling)

    def fill_(self, **kwargs):
        pass

    def compute_(self, **kwargs):
        question_decoded = self.dataset.question_tokenizer.decode(kwargs["state"].text.numpy()[0],
                                                                  ignored=["<SOS>"], stop_at_end=True)
        ref_questions = kwargs["ref_questions_decoded"]
        score = meteor_score(references=ref_questions, hypothesis=question_decoded)
        self.metric.append(score)


class KurtosisMetric(Metric):
    def __init__(self, agent, train_test, env, trunc, sampling):
        Metric.__init__(self, agent, train_test, "kurtosis", "scalar", env, trunc, sampling)

    def fill_(self, **kwargs):
        dist = pd.Series(kwargs["dist"].probs.squeeze().cpu().numpy())
        self.measure.append(dist.kurtosis())

    def compute_(self, **kwargs):
        self.metric.append(np.mean(self.measure))


class PeakinessMetric(Metric):
    def __init__(self, agent, train_test, env, trunc, sampling):
        Metric.__init__(self, agent, train_test, "peakiness", "scalar", env, trunc, sampling)

    def fill_(self, **kwargs):
        sorted, indices = torch.sort(kwargs["dist"].probs.cpu(), descending=True)
        sum_10 = sorted[:, :10].sum().item()
        self.measure.append(sum_10)

    def compute_(self, **kwargs):
        self.metric.append(np.mean(self.measure))


metrics = {"return": Return, "valid_actions": VAMetric, "size_valid_actions": SizeVAMetric,
           "dialog": DialogMetric, "dialogimage": DialogImageMetric,
           "ppl_dialog_lm": PPLDialogfromLM, "ppl_dialog_lmext": PPLDialogfromLMExt, "bleu": BleuMetric,
           "sum_probs": SumProbsOverTruncated,
           "ttr": UniqueWordsMetric,
           "selfbleu": SelfBleuImageMetric, "language_score": None,
           "histogram_answers": HistogramOracle, "cider": CiderMetric, "meteor": MeteorMetric,
           "kurtosis": KurtosisMetric, "peakiness": PeakinessMetric,
           "oracle": None}
metrics_to_tensorboard = ["return", "size_valid_actions", "sum_probs_truncated", "lm_valid_actions", "ttr",
                          "action_probs_truncated", "valid_actions_episode", "ppl_dialog_lm", "ttr_question"]
