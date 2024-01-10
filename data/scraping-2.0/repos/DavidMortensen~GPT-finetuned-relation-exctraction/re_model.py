from os.path import dirname, join

from os import path

import fire
import random
import torch

import numpy as np
from tempfile import NamedTemporaryFile
from torch import nn

from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, classification_report

from model_pytorch import DoubleHeadModel, load_openai_pretrained_model, dotdict
from loss import ClassificationLossCompute
from opt import OpenAIAdam
from dataset_functions import DatasetFunctions
from text_utils import TextEncoder, LabelEncoder
from train_utils import predict, iter_data, iter_apply, persist_model, load_model
from logging_utils import ResultLogger


def _remove_label_direction(label):
    direction_suffix_start = label.find('(')
    if direction_suffix_start != -1:
        return label[:direction_suffix_start]
    else:
        return label


def _get_max_label_length(labels):
    return max([len(label) for label in labels])


def run_epoch(model, train, dev, test, compute_loss_fct, batch_size, device, epoch, label_encoder, logger,
              negative_label, log_with_id=True, verbose=False):
    print('-' * 100)

    indices_train, mask_train, labels_train, _, _ = train

    n_batches = len(indices_train) // batch_size

    current_loss: float = 0
    seen_sentences = 0
    modulo = max(1, int(n_batches / 10))

    positive_labels = set(label_encoder.get_items())
    positive_labels.discard(negative_label)
    positive_labels = [label_encoder.get_idx_for_item(label) for label in positive_labels]

    epoch_labels_pred_train = []
    epoch_labels_train = []

    for batch_no, (batch_indices, batch_mask, batch_labels) in enumerate(iter_data(
        *shuffle(indices_train, mask_train, labels_train, random_state=np.random),
                 batch_size=batch_size, truncate=True, verbose=True)):

        model.train()
        
        x = torch.tensor(batch_indices, dtype=torch.long).to(device)
        y = torch.tensor(batch_labels, dtype=torch.long).to(device)
        mask = torch.tensor(batch_mask).to(device)

        lm_logits, clf_logits = model(x)
        loss = compute_loss_fct(x, y, mask, clf_logits, lm_logits)

        epoch_labels_pred_train.extend(np.argmax(clf_logits.detach().cpu(), 1))
        epoch_labels_train.extend(batch_labels)

        seen_sentences += len(batch_indices)
        current_loss += loss

        if batch_no % modulo == 0:
            train_acc = accuracy_score(epoch_labels_train, epoch_labels_pred_train) * 100
            train_micro_f1 = f1_score(epoch_labels_train, epoch_labels_pred_train, average='micro', labels=positive_labels)
            train_macro_f1 = f1_score(epoch_labels_train, epoch_labels_pred_train, average='macro', labels=positive_labels)
            print("epoch {0} - iter {1}/{2} - loss {3:.8f} - acc {4:.2f} - micro f1 {5:.2f} - macro f1 {6:.2f}"
                  .format(epoch, batch_no, n_batches, current_loss / seen_sentences, train_acc, train_micro_f1, train_macro_f1))

    current_loss /= len(indices_train)

    # IMPORTANT: Switch to eval mode
    model.eval()

    indices_dev, mask_dev, labels_dev, ids_dev, _ = dev

    print('-' * 100)
    dev_logits, dev_loss = iter_apply(indices_dev, mask_dev, labels_dev, model, compute_loss_fct, device, batch_size)

    avg_dev_loss = dev_loss / len(indices_dev)

    label_pred_dev = np.argmax(dev_logits, 1)

    dev_accuracy = accuracy_score(labels_dev, label_pred_dev) * 100.
    dev_micro_f1 = f1_score(labels_dev, label_pred_dev, average = 'micro', labels = positive_labels)
    dev_macro_f1 = f1_score(labels_dev, label_pred_dev, average = 'macro', labels = positive_labels)

    print('EVALUATION: cost: {} | acc: {} | micro f1: {} | macro f1: {}'.format(
        dev_loss / len(indices_dev), dev_accuracy, dev_micro_f1, dev_macro_f1))

    # save predictions on test dataset per epoch
    logger.log(train_loss=current_loss,
               dev_loss=avg_dev_loss,
               dev_accuracy=dev_accuracy,
               dev_micro_f1=dev_micro_f1,
               dev_macro_f1=dev_macro_f1)

    label_idxs_pred_dev, _ = predict(indices_dev, model, device, batch_size)
    labels_pred_dev = [label_encoder.get_item_for_index(label_index) for label_index in label_idxs_pred_dev]
    logger.log_dev_predictions(epoch, labels_pred_dev, ids_dev, log_with_id=log_with_id)

    
    indices_test, _, labels_test, ids_test, entity_ids_test = test

    log_pr_curve = len(labels_test) > 0 and entity_ids_test is not None

    label_idxs_pred_test, probs_test = predict(indices_test, model, device, batch_size,
                                                compute_probs=log_pr_curve)

    labels_pred_test = [label_encoder.get_item_for_index(label_index) for label_index in label_idxs_pred_test]
    logger.log_test_predictions(epoch, labels_pred_test, ids_test, log_with_id=log_with_id)

    if log_pr_curve:
        negative_label_idx = label_encoder.get_idx_for_item(negative_label)
        logger.log_test_pr_curve(epoch, entity_ids_test, labels_test, probs_test, negative_label_idx, label_encoder)

    return avg_dev_loss, dev_micro_f1, dev_macro_f1


def train(data_dir, log_dir, max_grad_norm=1, learning_rate=6.25e-5, learning_rate_warmup=0.002,
          n_ctx=512, n_embd=768, n_head=12, n_layer=12, embd_pdrop=.1, lm_coef=.5,
          attn_pdrop=.1, resid_pdrop=.1, clf_pdrop=.1, word_pdrop=.0, l2=0.01, vector_l2=True,
          optimizer='adam', afn='gelu', learning_rate_schedule='warmup_linear',
          encoder_path='model/encoder_bpe_40000.json', bpe_path='model/vocab_40000.bpe', n_transfer=12,
          beta1=.9, beta2=.999, e=1e-8, batch_size=8, max_epochs=3, dev_size=.1, seed=0,
          subsampling_rate=1.0, train_set_limit=None, dev_file=None, dev_set_limit=None, skip_test_set=False,
          verbose_fetcher=False, verbose_training=False, masking_mode=None, write_model=True):

    """
    n_ctx: Max sequence length
    """

    cfg = dotdict(locals().items())
    logger = ResultLogger(log_dir, **cfg)

    #Set Seeds to ensure uniform results
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    #GFX/Cuda Settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print('Device: {} | n_gpu: {}'.format(device, n_gpu))

    # create / load encoders from Language Model for text and labels, to ensure correct token encoding. 
    text_encoder = TextEncoder(encoder_path, bpe_path)
    label_encoder = LabelEncoder(add_unk=False)

    encoder = text_encoder.encoder
    encoder['_start_'] = len(encoder)
    encoder['_delimiter_'] = len(encoder)
    encoder['_delimiter2_'] = len(encoder)
    encoder['_classify_'] = len(encoder)
    n_special = 4


    for t in DatasetFunctions.MASKED_ENTITY_TOKENS:
        text_encoder.encoder[t] = len(text_encoder.encoder)
        n_special += 1

    # TODO: improve (as a sentence is generally much longer than the two entities)
    # the input has 3 parts (entity 1, entity 2, sentence) and special tokens
    # all together should not exceed the context length
    max_len = (n_ctx - n_special - 1) // 3

    #Load corpus from input data
    corpus = DatasetFunctions.fetch(data_dir, None, seed,
                                    negative_label = 'no_relation',
                                    subsampling_rate=subsampling_rate,
                                    train_set_limit=train_set_limit,
                                    dev_set_limit=dev_set_limit,
                                    skip_test_set=skip_test_set,
                                    predefined_dev_set = True,
                                    verbose=verbose_fetcher,
                                    masking_mode=masking_mode,
                                    dev_file=dev_file)

    #Encode corpus
    corpus = DatasetFunctions.encode(*corpus, text_encoder=text_encoder, label_encoder=label_encoder)

    #Redefine n_ctx based on lenght of tokens in corpus
    n_ctx = min(DatasetFunctions.max_length(*corpus, max_len=max_len) + n_special + 1, n_ctx)

    #Transform corpus into desired format
    transformed_corpus = DatasetFunctions.transform(*corpus, text_encoder=text_encoder, max_length=max_len, n_ctx=n_ctx)
    train, dev, test = transformed_corpus
    _, _, labels_dev, ids_dev, _ = dev
    
    logger.log_dev_labels(
        labels_dev=[label_encoder.get_item_for_index(label) for label in labels_dev],
        ids=ids_dev)

    batch_size_train = batch_size * max(n_gpu, 1)
    n_updates_total = (len(train[0]) // batch_size_train) * max_epochs

    clf_token = text_encoder.encoder['_classify_']
    vocab = len(text_encoder.encoder) + n_ctx
    n_class = len(label_encoder)
    dh_model = DoubleHeadModel(cfg, clf_token, ('classification', n_class), vocab, n_ctx)

    criterion = nn.CrossEntropyLoss(reduce=False)
    model_opt = OpenAIAdam(dh_model.parameters(),
                           lr=learning_rate,
                           schedule=learning_rate_schedule,
                           warmup=learning_rate_warmup,
                           t_total=n_updates_total,
                           b1=beta1,
                           b2=beta2,
                           e=e,
                           l2=l2,
                           vector_l2=vector_l2,
                           max_grad_norm=max_grad_norm)

    compute_loss_fct = ClassificationLossCompute(criterion,
                                                 criterion,
                                                 lm_coef,
                                                 model_opt)

    load_openai_pretrained_model(dh_model.transformer, n_ctx = n_ctx, n_special = n_special, n_transfer = n_transfer)

    dh_model.to(device)
    dh_model = nn.DataParallel(dh_model)

    if write_model:
        model_dir = path.join(logger.get_base_dir(), 'models')
        persist_model(model_dir, dh_model, text_encoder, label_encoder)

    #run training!
    best_f1 = 0.
    for epoch in range(1, max_epochs + 1):
        print(epoch)
        dev_loss, _, dev_macro_f1 = run_epoch(dh_model, train, dev, test, compute_loss_fct, batch_size, device, epoch,
                                              label_encoder, logger, negative_label = 'no_relation',
                                              log_with_id = False, verbose=verbose_training)
        if dev_macro_f1 > best_f1:
            best_f1 = dev_macro_f1

            if write_model:
                print(f'Saving model at epoch {epoch}. With dev_f1 score of {dev_macro_f1}.')
                model_file_name = f'model_epoch-{epoch}_dev-macro-f1-{dev_macro_f1}_' \
                                  f'dev-loss-{dev_loss}_{logger.start_time}.pt'
                persist_model(model_dir, dh_model, text_encoder, label_encoder, model_name=model_file_name)


def evaluate(test_file, log_dir, save_dir, model_file = 'model.pt', batch_size = 8, masking_mode = None):
    cfg = dotdict(locals().items())
    print(cfg)

    logger = ResultLogger(log_dir, **cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, text_encoder, label_encoder = load_model(save_dir, model_file=model_file)

    model = model.to(device)

    n_ctx = model.n_ctx
    max_len = 512 // 3

    test = DatasetFunctions._load_from_jsonl(test_file, is_test=False, masking_mode=masking_mode)
    test = DatasetFunctions.encode(test, text_encoder=text_encoder, label_encoder=label_encoder)
    test = DatasetFunctions.transform(*test, text_encoder=text_encoder, max_length=max_len, n_ctx=n_ctx)[0]

    negative_label = 'no_relation'

    indices_test, _, label_idxs_test, ids_test, entity_ids_test = test

    log_pr_curve = entity_ids_test is not None

    label_idxs_pred, probs_test = predict(indices_test, model, device, batch_size, compute_probs=log_pr_curve)
    labels_pred_test = [label_encoder.get_item_for_index(label_index) for label_index in label_idxs_pred]
    logger.log_test_predictions(0, labels_pred_test, ids_test)

    test_accuracy = accuracy_score(label_idxs_test, label_idxs_pred) * 100.
    
    labels = list(sorted(set(label_idxs_test)))
    labels.remove(label_encoder.get_idx_for_item(negative_label))

    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
        label_idxs_test, label_idxs_pred, average='micro', labels=labels)
    print(f'TEST: ACC: {test_accuracy} | P: {test_precision} | R: {test_recall} | F1: {test_f1}')

    if log_pr_curve:
        negative_label_idx = label_encoder.get_idx_for_item(negative_label)
        logger.log_test_pr_curve(0, entity_ids_test, label_idxs_test, probs_test, negative_label_idx, label_encoder)

    logger.close()


if __name__ == '__main__':
    fire.Fire({
        'train': train,
        'evaluate': evaluate
    })