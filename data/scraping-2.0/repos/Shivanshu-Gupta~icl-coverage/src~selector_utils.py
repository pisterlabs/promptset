import torch
from params import AllParams
from constants import Dataset as D, ExSel as ES

def get_selector(
    P: AllParams, candidates, example_template, test_ds,
    enc_len_fn=None, max_len=-1, subtract_gen_len=False,
):
    SP = P.selector
    if SP.selector_type in [
        ES.COSINE, ES.STRUCT, ES.BERTSCORE, ES.LF_COVERAGE
    ]:
        from selector import BertScoreSelector, CosineCoverageSelector, StructuralCoverageSelector, LFCoverageSelector
        selector_params = P.selector
        common_args = dict(args=selector_params,
                           examples=candidates,
                           example_template=example_template)
        device = f"cuda:{P.gpu}" if torch.cuda.is_available() and P.gpu >= 0 else "cpu"
        if SP.selector_type == ES.COSINE:
            ex_selector = CosineCoverageSelector.from_examples(
                **common_args, query_examples=test_ds, device=device,
                enc_len_fn=enc_len_fn, max_len=max_len,
                subtract_gen_len=subtract_gen_len,)
        elif SP.selector_type == ES.STRUCT:
            ex_selector = StructuralCoverageSelector.from_examples(
                **common_args, query_examples=test_ds,
                enc_len_fn=enc_len_fn, max_len=max_len,
                subtract_gen_len=subtract_gen_len,)
        elif SP.selector_type == ES.BERTSCORE:
            ex_selector = BertScoreSelector.from_examples(
                **common_args, query_examples=test_ds, device=device,
                enc_len_fn=enc_len_fn, max_len=max_len,
                subtract_gen_len=subtract_gen_len)
        elif SP.selector_type == ES.LF_COVERAGE:
            ex_selector = LFCoverageSelector.from_examples(
                **common_args, query_examples=test_ds,
                enc_len_fn=enc_len_fn, max_len=max_len,
                subtract_gen_len=subtract_gen_len)
    else:
        raise ValueError(f'Unknown selector type: {SP.selector_type}')
    return ex_selector

if __name__ == '__main__':
    from pathlib import Path
    from functools import partial
    from langchain.prompts import FewShotPromptTemplate2
    from data_utils import get_dataset, get_templates
    from constants import Dataset as D, ExSel as ES, max_new_tokens_d, context_length_limit, default_prompt_version, LLM
    from tools.lm import get_enc_len_fn
    dataset = D.QNLI
    lm_name = LLM.NEO
    # lm_name = LLM.LLAMA13B
    input_feature, train_split, test_split = {
        D.SMCALFLOW_CS: ('source', 'train', 'comp_test'),
        D.GEOQUERY: ('source', 'template_1_train', 'template_1_test'),
        D.OVERNIGHT: ('paraphrase', 'socialnetwork_template_0_train', 'socialnetwork_template_0_test'),
        D.MTOP: (None, 'train', 'validation'),
        D.BREAK: ('question_text', 'validation', 'validation'),
        D.DROP: ('question', 'train', 'validation'),
        D.QNLI: ('', 'train', 'validation'),
        D.GSM8K: ('', 'train', 'test'),
        D.AQUA: ('', 'train', 'validation'),
    }[dataset]
    print('Dataset'.center(80, '='))
    ds = get_dataset(dataset, Path('../data'), None)
    candidates = ds[train_split]
    test_ds = ds[test_split].select(range(0, 10))
    test_ex = test_ds[0]

    print(ds)

    print('Example Template'.center(80, '='))
    templates = get_templates(dataset, default_prompt_version[lm_name], input_feature=input_feature)
    example_template = templates['example_template']
    print(example_template.format(**candidates[0]))
    print('Fewshot Prompt'.center(80, '='))
    fewshot_prompt_fn = partial(FewShotPromptTemplate2,
        input_variables=templates['example_template'].input_variables,
        example_separator='\n\n', **templates)
    print(fewshot_prompt_fn(examples=list(candidates.select(range(4)))).format(**test_ex))

    max_len = context_length_limit[lm_name] - max_new_tokens_d[dataset]
    enc_len_fn = get_enc_len_fn(lm_name)
    generation_kwargs = dict(do_sample=False, max_new_tokens=256)
    from langchain.llms.huggingface import HuggingFace
    llm = HuggingFace.from_model_name(
        lm_name.value, device=0, task='text-generation', batch_size=4,
        generation_kwargs=generation_kwargs)


    n_shots = -1
    common_args = dict(dataset=dataset, n_shots=50 if n_shots == -1 else n_shots, gpu=0)
    selector_args = {
        # 'similar': dict(selector_type=ES.SIMILAR, sim_metric=['L2', 'IP'][0]),
        # 'cosine': dict(selector_type=ES.COSINE, coverage=False),
        # 'cosine_coverage': dict(selector_type=ES.COSINE, coverage=True, reorder=True),
        # 'recall': dict(selector_type=ES.STRUCT,
        #     substruct='depst', subst_size=4, depparser='spacy',
        #     selector_metric='recall', coverage=False,),
        # 'recall_coverage': dict(selector_type=ES.STRUCT,
        #     substruct='depst', subst_size=4, depparser='spacy',
        #     selector_metric='recall', coverage=True, ordering='recall'),
        # 'bm25_ngram': dict(selector_type=ES.STRUCT,
        #     substruct='ngram', subst_size=1,
        #     selector_metric='bm25', coverage=False),
        # 'bm25_depst': dict(selector_type=ES.STRUCT,
        #     substruct='depst', subst_size=4, depparser='spacy',
        #     selector_metric='bm25', coverage=False),
        # 'bm25_coverage': dict(selector_type=ES.STRUCT,
        #     substruct='depst', subst_size=4, depparser='spacy',
        #     selector_metric='bm25', coverage=True, ordering='bm25'),
        'bertscore': dict(selector_type=ES.BERTSCORE, selector_metric='recall',
            emb_lm='microsoft/deberta-base-mnli', idf=True, coverage=False),
        # 'bertscore_coverage': dict(selector_type=ES.BERTSCORE, selector_metric='recall',
        #     emb_lm='microsoft/deberta-base-mnli', idf=True, coverage=True, ordering='recall'),
    }
    for selector in selector_args:
        print(selector.center(80, '='))
        P = AllParams(**common_args, **selector_args[selector])
        ex_selector = get_selector(P, candidates, templates['example_template'], test_ds)
        fewshot_prompt = fewshot_prompt_fn(
            example_selector=ex_selector, max_len=max_len, enc_len_fn=enc_len_fn)

        print('example'.center(80, '-'))
        print(test_ex)

        print('template'.center(80, '-'))
        print(example_template.format(**test_ex))

        if selector != 'similar':
            shots, scores = ex_selector.select_examples(test_ex, return_scores=True)
            print('scores'.center(80, '-'))
            print(scores)
        else:
            shots = ex_selector.select_examples(test_ex)

        print('prompt'.center(80, '-'))
        score = None
        prompt = fewshot_prompt.format_from_examples(shots, **test_ex)
        print(prompt)

        print('-' * 80)

    if False: # DROP embed context dev scratch
        sents = ['\n'.join([ex['passage'], ex['question']]) for ex in candidates]
        from bert_score.utils import get_idf_dict, sent_encode, get_bert_embedding, get_tokenizer, model2layers, get_model
        import numpy as np
        from collections import defaultdict
        from tools.track import track
        model = get_model(P.emb_lm, model2layers[P.emb_lm]).eval()
        tokenizer = get_tokenizer(P.emb_lm, use_fast=False)
        idf_dict = defaultdict(lambda: 1.0)
        # set idf for [SEP] and [CLS] to 0
        idf_dict[tokenizer.sep_token_id] = 0
        idf_dict[tokenizer.cls_token_id] = 0
        from bert_score.utils import get_bert_embedding
        device = 'cpu'
        embs0, masks0, padded_idf0 = get_bert_embedding(
            [ds['train'][0]['question']], model, tokenizer, idf_dict, device=device, all_layers=False
        )
        embs1, masks1, padded_idf1 = get_bert_embedding(
            sents[:1], model, tokenizer, idf_dict, device=device, all_layers=False
        )

        batch_size = 64
        embs_l, idfs_l = [], []
        for batch_start in track(range(0, len(sents), batch_size)):
            sen_batch = sents[batch_start:batch_start+batch_size]
            embs, masks, padded_idf = get_bert_embedding(
                sen_batch, model, tokenizer, idf_dict, device=device, all_layers=False
            )
            embs = embs.cpu().numpy()
            masks = masks.cpu().numpy()
            padded_idf = padded_idf.cpu().numpy()
            for i in range(len(sen_batch)):
                sequence_len = masks[i].sum().item()
                emb = embs[i, :sequence_len]
                emb /= np.linalg.norm(emb, axis=1, keepdims=True)
                idf = padded_idf[i, :sequence_len]
                embs_l.append(emb)
                idfs_l.append(idf)