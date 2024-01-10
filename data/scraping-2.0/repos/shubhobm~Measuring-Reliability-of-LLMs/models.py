import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification,\
     T5Tokenizer, T5ForConditionalGeneration, set_seed

# import json, jsonlines, sacrebleu, logging, copy
from tqdm import tqdm
# from torchseq.agents.para_agent import ParaphraseAgent
# from torchseq.datasets.json_loader import JsonDataLoader
# from torchseq.utils.config import Config
import torch
from openai_api_key import *
import openai

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LLM():
    def __init__(self, model_path="gpt2", max_len=50, seed=42):
        super(LLM, self).__init__()
        self.max_len = max_len
        self.seed = 42
        self.template = """
        Passage: {}
        Question: {}
        Answer:"""
        self.generator = pipeline('text-generation', model=model_path)


    def generate(self, data_bs):
        '''
        Inputs:
            -data_bs (List(dict)) : passage, question
        '''
        set_seed(self.seed)

        prompt_bs = [self.template.format(data_bs[i]["passage"], data_bs[i]["question"]) for i in range(len(data_bs))]

        out_bs = self.generator(prompt_bs, max_length=50)
        if isinstance(out_bs[0], dict):
            gen_bs = [out_bs[i]['generated_text'].replace(prompt_bs[i], '') for i in range(len(out_bs))]
        else:
            gen_bs = [out_bs[i][0]['generated_text'].replace(prompt_bs[i], '') for i in range(len(out_bs))]

        return gen_bs


class Cond_PP():
    def __init__(self, tok_path="t5-base", model_path="coderpotter/T5-for-Adversarial-Paraphrasing", max_len=30, template= "paraphrase: {} </s>"):
        super(Cond_PP, self).__init__()
        self.paraphrasing_tokenizer = T5Tokenizer.from_pretrained(tok_path)
        self.paraphrasing_model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
        self.template = template

    # this function will take a sentence, top_k and top_p values based on beam search
    def generate_pp(self, sentence, num_pp=3, top_k=120, top_p=0.95, sample=True, beam_search=True):
        text = self.template.format(sentence)
        encoding = self.paraphrasing_tokenizer.encode_plus(text, max_length=256, padding="max_length", return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"], encoding["attention_mask"]
        beam_outputs = self.paraphrasing_model.generate(
            input_ids=input_ids.to(device),
            attention_mask=attention_masks.to(device),
            do_sample=sample,
            max_length=256,
            top_k=top_k,
            top_p=top_p,
            early_stopping=True,
            num_return_sequences=num_pp,
            num_beams=num_pp if beam_search else 1
        )
        final_outputs = []
        for beam_output in beam_outputs:
            sent = self.paraphrasing_tokenizer.decode(beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            if sent.lower() != sentence.lower() and sent not in final_outputs:
                final_outputs.append(sent)
        return final_outputs


class Separator_PP():
    def __init__(self, path_to_model='./models/separator-wa-v1.2/', data_path='./data/'):
        super(Separator_PP, self).__init__()
        checkpoint_path = path_to_model + "model/checkpoint.pt"

        # prepare configs
        with open(path_to_model + "config.json") as f:
            cfg_dict = json.load(f)
        cfg_dict["env"]["data_path"] = data_path
        cfg_dict["dataset"] = "json"
        cfg_dict["json_dataset"] = {
            "path": None,
            "field_map": [
                {"type": "copy", "from": "input", "to": "s2"},
                {"type": "copy", "from": "exemplar", "to": "template"},
                {"type": "copy", "from": "input", "to": "s1"},
            ],
        }
        self.config = Config(cfg_dict)
        self.instance = ParaphraseAgent(config=self.config, run_id=None, output_path="./runs/parademo/", silent=True, verbose=False, training_mode=False)
        self.instance.load_checkpoint(checkpoint_path)

    def generate_pp(self, texts):
        data_loader = JsonDataLoader(self.config, test_samples=texts)
        self.instance.model.eval()

        loss, metrics, (pred_output, gold_output, gold_input), memory_values_to_return = self.instance.inference(data_loader.test_loader)

        return pred_output

class HRQVAE_PP():
    def __init__(self, path_to_model='./models/hrqvae_paralex/', data_path='./data/', top_k=3):
        super(HRQVAE_PP, self).__init__()

        self.top_k = top_k

        # Create the dataset and model
        with open(path_to_model + "config.json") as f:
            cfg_dict = json.load(f)

        self.logger = logging.getLogger('TorchSeq')

        self.config = Config(cfg_dict)
        checkpoint_path = path_to_model + "model/checkpoint.pt"
        self.agent = ParaphraseAgent(config=self.config, run_id=None,  output_path=None, data_path=data_path, silent=False, verbose=False, training_mode=False)

        # Load the checkpoint
        self.agent.load_checkpoint(checkpoint_path)

    def generate_pp(self, texts):
        texts = [{'sem_input': t} for t in texts]
        self.agent.model.eval()

        # Generate encodings
        self.logger.info("Generating encodings for eval set")
        config_gen_eval = copy.deepcopy(self.config.data)
        config_gen_eval["dataset"] = "json"
        config_gen_eval["json_dataset"] = {
            "path": self.config.eval.metrics.sep_ae.eval_dataset,
            "field_map": [
                {"type": "copy", "from": "sem_input", "to": "s2"},
                {"type": "copy", "from": "sem_input", "to": "template"},
                {"type": "copy", "from": "sem_input", "to": "s1"},
            ],
        }
        config_gen_eval["eval"]["topk"] = 1

        data_loader = JsonDataLoader(
            data_path=self.agent.data_path,
            config=Config(config_gen_eval),
            dev_samples=texts,
        )

        self.config.eval.data["sample_outputs"] = False

        post_bottleneck = (
            "_after_bottleneck" if self.agent.config.bottleneck.code_predictor.get("post_bottleneck", False) else ""
        )

        _, _, (_, _, _), memory_eval = self.agent.inference(
            data_loader.valid_loader,
            memory_keys_to_return=[f"sep_encoding_1{post_bottleneck}", f"sep_encoding_2{post_bottleneck}", "vq_codes"],
        )

        if not self.agent.config.bottleneck.code_predictor.get("sem_only", False):
            X_eval = torch.cat(
                [
                    memory_eval[f"sep_encoding_1{post_bottleneck}"][:, 0, :],
                    memory_eval[f"sep_encoding_2{post_bottleneck}"][:, 0, :],
                ],
                dim=1,
            )
        else:
            X_eval = memory_eval[f"sep_encoding_1{post_bottleneck}"][:, 0, :]
        y_eval = memory_eval["vq_codes"]

        # Get top-k predicted codes

        self.logger.info("Running code predictor")

        if self.agent.model.code_predictor.config.get("beam_width", 0) < self.top_k:
            self.agent.model.code_predictor.config.data["beam_width"] = self.top_k

        pred_codes = []
        # # TODO: batchify!
        for ix, x_batch in enumerate(tqdm(X_eval, desc="Predicting codes")):
            curr_codes = self.agent.model.code_predictor.infer(
                x_batch.unsqueeze(0).to(self.agent.device), {}, outputs_to_block=y_eval[ix].unsqueeze(0), top_k=self.top_k
            )
            pred_codes.append(curr_codes)


        pred_codes = torch.cat(pred_codes, dim=0)

        config_pred_diversity = copy.deepcopy(self.config.data)
        config_pred_diversity["dataset"] = "json"
        config_pred_diversity["json_dataset"] = {
            "path": self.config.eval.metrics.sep_ae.eval_dataset,
            "field_map": [
                {"type": "copy", "from": "sem_input", "to": "s2"},
                {"type": "copy", "from": "sem_input", "to": "template"},
                {"type": "copy", "from": "sem_input", "to": "s1"},
                {"type": "copy", "from": "forced_codes", "to": "forced_codes"},
            ],
        }
        config_pred_diversity["eval"]["topk"] = 1

        data_loader = JsonDataLoader(
            data_path=self.agent.data_path,
            config=Config(config_pred_diversity),
            dev_samples=texts,
        )

        self.config.eval.data["sample_outputs"] = True

        topk_outputs = []

        for k in range(self.top_k):
            self.logger.info(f"Running generation with {k+1}th best codes")

            samples = data_loader._valid.samples
            samples = [{**x, "forced_codes": pred_codes[i, k, :].tolist()} for i, x in enumerate(samples)]
            forced_loader = JsonDataLoader(
                data_path=self.agent.data_path, config=Config(config_pred_diversity), dev_samples=samples
            )

            _, _, (output, _, _), _ = self.agent.inference(forced_loader.valid_loader)


            topk_outputs.append(output)
        return topk_outputs

class QualityControl_PP():
    def __init__(self, type):
        super(QualityControl_PP, self).__init__()
        assert type in ['captions', 'questions', 'sentences']
        self.pipe = pipeline('text2text-generation', model=f'ibm/qcpg-{type}', device=0)
        self.ranges = {
            'captions': {'lex': [0, 90], 'syn': [0, 80], 'sem': [0, 95]},
            'sentences': {'lex': [0, 100], 'syn': [0, 80], 'sem': [0, 95]},
            'questions': {'lex': [0, 90], 'syn': [0, 75], 'sem': [0, 95]}
        }[type]

    def generate_pp(self, text, lexical, syntactic, semantic, **kwargs):
        assert all([0 <= val <= 1 for val in [lexical, syntactic, semantic]]), \
                 f' control values must be between 0 and 1, got {lexical}, {syntactic}, {semantic}'
        names = ['semantic_sim', 'lexical_div', 'syntactic_div']
        control = [int(5 * round(val * 100 / 5)) for val in [semantic, lexical, syntactic]]
        control ={name: max(min(val , self.ranges[name[:3]][1]), self.ranges[name[:3]][0]) for name, val in zip(names, control)}
        control = [f'COND_{name.upper()}_{control[name]}' for name in names]
        assert all(cond in self.pipe.tokenizer.additional_special_tokens for cond in control)
        text = ' '.join(control) + text if isinstance(text, str) else [' '.join(control) for t in text]
        return self.pipe(text, **kwargs)


class LLM_PP():
    def __init__(self, engine="text-ada-001", temp=0.8, max_tokens=120,
             top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0):
        super(LLM_PP, self).__init__()
        openai.api_key = key
        self.engine = engine
        self.temp = temp
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.template_head = """Generate diverse paraphrases taking motivation from the examples given below."""
        self.template_body = """

        Sentence :{}
        Paraphrase :{}
        """
    def generate_pp(self, exemplars_pair, text, temperature=1.0, top_p=1.0, num_pp=3,):
        prompt = self.template_head
        for e in exemplars_pair:
            prompt += self.template_body.format(e[0], e[1])
        prompt += f"""

        Sentence :{text}
        Paraphrase :"""

        response = openai.Completion.create(
          engine=self.engine,
          prompt=prompt,
          temperature=temperature,
          max_tokens=self.max_tokens,
          top_p=top_p,
          frequency_penalty=self.frequency_penalty,
          presence_penalty=self.presence_penalty,
        )
        outs = [response['choices'][i]['text'] for i in range(min(num_pp, len(response['choices'])))]
        return outs


class PP_Detector():
    def __init__(self, tok_path="domenicrosati/deberta-v3-large-finetuned-paws-paraphrase-detector", model_path="domenicrosati/deberta-v3-large-finetuned-paws-paraphrase-detector", max_len=30):
        super(PP_Detector, self).__init__()
        self.detection_tokenizer = AutoTokenizer.from_pretrained(tok_path)
        self.detection_model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.detection_model.to(device)

    def score_binary(self, y_1, y_2):
        inputs = self.detection_tokenizer(y_1, y_2, return_tensors="pt", padding=True).to(device)
        outputs = self.detection_model(**inputs)
        scores = outputs.logits.softmax(dim=-1)
        # Return probabilites and scores for not paraphrase and paraphrase
        return scores.T[0].item(), scores.T[1].item()


class NLI():
    """
    microsoft/deberta-v2-xxlarge-mnli uses
    "id2label": {
        "0": "CONTRADICTION",
        "1": "NEUTRAL",
        "2": "ENTAILMENT"
      },
    """
    def __init__(self, tok_path="microsoft/deberta-base-mnli", model_path="microsoft/deberta-base-mnli", max_len=30):
        super(NLI, self).__init__()
        self.detection_tokenizer = AutoTokenizer.from_pretrained(tok_path)
        self.detection_model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.detection_model.to(device)

    def entailed(self, y_1, y_2):
        inputs = self.detection_tokenizer(y_1, y_2, return_tensors="pt", padding=True).to(device)
        outputs = self.detection_model(**inputs)
        scores = outputs.logits.softmax(dim=-1)
        return scores.T[2].item()

    def contradicted(self, y_1, y_2):
        inputs = self.detection_tokenizer(y_1, y_2, return_tensors="pt", padding=True).to(device)
        outputs = self.detection_model(**inputs)
        scores = outputs.logits.softmax(dim=-1)
        return scores.T[0].item()