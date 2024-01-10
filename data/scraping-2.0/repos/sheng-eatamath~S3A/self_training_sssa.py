import os
import math
import sys
import pickle
import json
from copy import deepcopy
from typing import Iterable
import numpy as np
import random
import itertools
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import reduce
import utils
from timm.utils import accuracy
from my_util_package.evaluate import cluster_acc
from nltk.corpus import wordnet as wn
from torchmetrics import Accuracy
from tqdm import tqdm
from collections import Counter, defaultdict, OrderedDict
from config import PROJECT_HOME

def train_one_epoch(model: torch.nn.Module, args, train_config,
                    data_loader, optimizer, amp_autocast,
                    device, epoch, loss_scaler, 
                    log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, num_training_steps_per_epoch=None, 
                    model_ema=None, other_params={}):
    module_train(model)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    print('train_one_epoch')
    for step, batch_data in enumerate(metric_logger.log_every(data_loader, print_freq, header, total_iter=num_training_steps_per_epoch)):
        
        loss = 0.0
        ((images_weak, images_strong), targets, targets_clu, img_idx, pred_cluster_targets) = batch_data[:5]
        # assign learning rate for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]

        # ramp-up ema decay 
        model_ema.decay = train_config['model_ema_decay_init'] + (args.model_ema_decay - train_config['model_ema_decay_init']) * min(1, it/train_config['warm_it'])
        metric_logger.update(ema_decay=model_ema.decay)
        
        ### data moving
        use_nonblocking = False
        images_weak, images_strong = images_weak.to(device, non_blocking=use_nonblocking), images_strong.to(device, non_blocking=use_nonblocking)
        pred_cluster_targets = pred_cluster_targets.to(device, non_blocking=use_nonblocking)
        if not args.oov_dataset:
            targets = targets.to(device, non_blocking=use_nonblocking)

        ### pseudo labeling
        with torch.no_grad():
            # pseudo-label with ema model
            logits_ema, prj_cls_ema, prj_features_ema = model_ema.ema(images_weak, return_features=True)
            probs_ema = F.softmax(logits_ema, dim=-1)
            score, pseudo_targets = probs_ema.max(-1)
                
            conf_mask = score>train_config['conf_threshold']
            if args.oov_dataset:
                pseudo_label_acc = 0
            else:
                pseudo_label_acc = (pseudo_targets[conf_mask] == targets[conf_mask]).float().mean().item() if conf_mask.sum().item() > 0 else 0
            conf_ratio = conf_mask.float().sum()/conf_mask.size(0) if conf_mask.sum().item() > 0 else 0
            metric_logger.update(conf_ratio=conf_ratio)
            metric_logger.update(pseudo_label_acc=pseudo_label_acc)
                
        
        with amp_autocast():
            logits, prj_cls, prj_features = model(
                images_strong, 
                return_features=True)
                
            # self-training loss
            loss_st = F.cross_entropy(logits[conf_mask], pseudo_targets[conf_mask]) if conf_mask.sum().item() > 0 else torch.tensor(0.0)
            loss_cluster = F.cross_entropy(logits, pred_cluster_targets)
            
            w_str = args.w_str
            w_ins = args.w_ins
            loss += w_ins * loss_st + w_str * loss_cluster
        
        loss_value = loss.item()
        
        
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            import pdb; pdb.set_trace()

        optimizer.zero_grad()
        if loss_scaler is not None:
            grad_norm = loss_scaler(loss, optimizer, clip_grad=None, parameters=model.parameters(), create_graph=False)
            loss_scale_value = loss_scaler.state_dict()["scale"]
            metric_logger.update(loss_scale=loss_scale_value)
            metric_logger.update(grad_norm=grad_norm)
        else:                   
            loss.backward(create_graph=False)
            optimizer.step()


        model_ema.update(model)
        metric_logger.update(loss_st=loss_st.item())
        
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)

        if log_writer is not None:            
            log_writer.update(loss_st=loss_st.item(), head="train")
            log_writer.update(loss_cluster=loss_cluster.item(), head="train")
            log_writer.update(conf_ratio=conf_ratio, head="train")
            log_writer.update(pseudo_label_acc=pseudo_label_acc, head="train")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
        
        ### control iteration number
        if (args.total_iter!=-1) and (step%num_training_steps_per_epoch==(num_training_steps_per_epoch-1)):
            break
        
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, model_ema=None, args=None, mask=None,
             other_params={}, log_writer=None):
    
    criterion = torch.nn.CrossEntropyLoss()
    acc_top3 = Accuracy(task='multiclass', num_classes=args.nb_classes, top_k=3).to(args.device)
    acc_top5 = Accuracy(task='multiclass', num_classes=args.nb_classes, top_k=5).to(args.device)
    acc_top10 = Accuracy(task='multiclass', num_classes=args.nb_classes, top_k=10).to(args.device)
    acc_top3_ema = Accuracy(task='multiclass', num_classes=args.nb_classes, top_k=3).to(args.device)
    acc_top5_ema = Accuracy(task='multiclass', num_classes=args.nb_classes, top_k=5).to(args.device)
    acc_top10_ema = Accuracy(task='multiclass', num_classes=args.nb_classes, top_k=10).to(args.device)

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'TrainVal:'

    # switch to evaluation mode
    module_eval(model)
    if model_ema is not None:
        module_eval(model_ema.ema)
    
    all_label_pred = []
    all_label_pred_ema = []
    all_label_target = []
    all_label_target_clu = []
    all_prj_features_ema = []
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0].to(device)
        target = batch[1].to(device)

        ### inference - student
        output, _ = model(images)
        acc = accuracy(output, target)[0]
        metric_logger.meters['acc1'].update(acc.item(), n=images.shape[0])
        metric_logger.meters['acc_top3'].update(acc_top3(output, target).item(), n=images.shape[0])
        metric_logger.meters['acc_top5'].update(acc_top5(output, target).item(), n=images.shape[0])
        metric_logger.meters['acc_top10'].update(acc_top10(output, target).item(), n=images.shape[0])
        
        ### inference - teacher
        ema_output, prj_features_ema = model_ema.ema(images) 
        ema_acc1 = accuracy(ema_output, target)[0]  
        metric_logger.meters['ema_acc1'].update(ema_acc1.item(), n=images.shape[0])
        metric_logger.meters['acc_top3_ema'].update(acc_top3_ema(ema_output, target).item(), n=images.shape[0])
        metric_logger.meters['acc_top5_ema'].update(acc_top5_ema(ema_output, target).item(), n=images.shape[0])
        metric_logger.meters['acc_top10_ema'].update(acc_top10_ema(ema_output, target).item(), n=images.shape[0])
        
        all_label_pred_ema.append(deepcopy(ema_output.argmax(dim=-1).cpu()))
        all_label_pred.append(deepcopy(output.argmax(dim=-1).cpu()))
        all_label_target.append(deepcopy(target.cpu()))
        all_label_target_clu.append(deepcopy(batch[2].cpu()))
        all_prj_features_ema.append(deepcopy(prj_features_ema.cpu().numpy()))

    all_label_pred = torch.cat(all_label_pred, dim=0).numpy()
    all_label_pred_ema = torch.cat(all_label_pred_ema, dim=0).numpy()
    all_label_target = torch.cat(all_label_target, dim=0).numpy()
    all_label_target_clu = torch.cat(all_label_target_clu, dim=0).numpy()
    all_prj_features_ema = np.concatenate(all_prj_features_ema)
    print('* Acc@1 {top1.global_avg:.3f}'.format(top1=metric_logger.acc1)) 
    print('* Acc@3 {top3.global_avg:.3f}'.format(top3=metric_logger.acc_top3)) 
    print('* Acc@5 {top5.global_avg:.3f}'.format(top5=metric_logger.acc_top5)) 
    print('* Acc@10 {top10.global_avg:.3f}'.format(top10=metric_logger.acc_top10)) 
    n_missing_labels = len(set(all_label_target.tolist()) - set(all_label_pred.tolist()))
    print(f'* Missing Label {n_missing_labels}')
    stats_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    stats_dict.update({'n_missing_labels': n_missing_labels})
    
    ### clustering acc
    print('evaluate:: cluster evaluate')
    mapping_ind_all_label_pred = dict(zip(np.unique(all_label_pred), range(np.unique(all_label_pred).shape[0])))
    all_label_pred_mapped = np.array(list(map(lambda x: mapping_ind_all_label_pred[x], all_label_pred.tolist())))
    acc_clu = cluster_acc(y_true=all_label_target_clu, y_pred=all_label_pred_mapped)
    stats_dict.update({'acc_clu': acc_clu})
    print('* acc_clu {acc_clu:.4f}'.format(acc_clu=acc_clu))
    mapping_ind_all_label_pred_ema = dict(zip(np.unique(all_label_pred_ema), range(np.unique(all_label_pred_ema).shape[0])))
    all_label_pred_ema_mapped = np.array(list(map(lambda x: mapping_ind_all_label_pred_ema[x], all_label_pred_ema.tolist())))
    acc_clu_ema = cluster_acc(y_true=all_label_target_clu, y_pred=all_label_pred_ema_mapped)
    stats_dict.update({'acc_clu_ema': acc_clu_ema})
    print('* acc_clu_ema {acc_clu_ema:.4f}'.format(acc_clu_ema=acc_clu_ema))
        
    vocab = other_params['vocab']
    mapping_idx_to_name = lambda x: vocab.mapping_idx_names[x]
    all_label_pred_ema_name = np.array([mapping_idx_to_name(x.item()) for x in all_label_pred_ema])
    all_label_pred_name = np.array([mapping_idx_to_name(x.item()) for x in all_label_pred])
    all_label_target_name = np.array([mapping_idx_to_name(x.item()) for x in all_label_target])
    score = measure_similarity_bert(vocab, all_label_pred_name, all_label_target_name, device=args.device)
    print(f'score={score}')
    score_ema = measure_similarity_bert(vocab, all_label_pred_ema_name, all_label_target_name, device=args.device)
    print(f'score={score_ema}')

    return stats_dict, all_prj_features_ema



from my_util_package.evaluate import measure_similarity_bert
@torch.no_grad()
def evaluate_label(data_loader, model, device, model_ema=None, args=None, mask=None,
             other_params={}, log_writer=None):
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'TrainVal:'
    vocab = other_params['vocab']
    # switch to evaluation mode
    module_eval(model)
    if model_ema is not None:
        module_eval(model_ema.ema)
    
    all_label_pred = []
    all_label_pred_ema = []
    all_label_target = []
    all_label_target_clu = []
    all_prj_features_ema = []
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0].to(device)
        target = batch[1]

        ### inference - student
        output, _ = model(images)
        ### inference - teacher
        ema_output, prj_features_ema = model_ema.ema(images) 

        all_label_pred_ema.append(deepcopy(ema_output.argmax(dim=-1).cpu()))
        all_label_pred.append(deepcopy(output.argmax(dim=-1).cpu()))
        all_label_target.extend(deepcopy(target))
        all_label_target_clu.append(deepcopy(batch[2].cpu()))
        all_prj_features_ema.append(deepcopy(prj_features_ema.cpu().numpy()))

    all_label_pred = torch.cat(all_label_pred, dim=0).numpy()
    all_label_pred_ema = torch.cat(all_label_pred_ema, dim=0).numpy()
    all_label_target_clu = torch.cat(all_label_target_clu, dim=0).numpy()
    all_prj_features_ema = np.concatenate(all_prj_features_ema)
    
    sim_bert = measure_similarity_bert(vocab, all_label_pred, all_label_target, device=device)
    sim_bert_ema = measure_similarity_bert(vocab, all_label_pred_ema, all_label_target, device=device)
    
    stats_dict = {
        'sim_bert': sim_bert['all-mpnet-base-v2'],
        'sim_bert_ema': sim_bert_ema['all-mpnet-base-v2'],
    }
    
    ### clustering acc
    print('evaluate:: cluster evaluate')
    mapping_ind_all_label_pred = dict(zip(np.unique(all_label_pred), range(np.unique(all_label_pred).shape[0])))
    all_label_pred_mapped = np.array(list(map(lambda x: mapping_ind_all_label_pred[x], all_label_pred.tolist())))
    acc_clu = cluster_acc(y_true=all_label_target_clu, y_pred=all_label_pred_mapped)
    stats_dict.update({'acc_clu': acc_clu})
    print('* acc_clu {acc_clu:.4f}'.format(acc_clu=acc_clu))
    mapping_ind_all_label_pred_ema = dict(zip(np.unique(all_label_pred_ema), range(np.unique(all_label_pred_ema).shape[0])))
    all_label_pred_ema_mapped = np.array(list(map(lambda x: mapping_ind_all_label_pred_ema[x], all_label_pred_ema.tolist())))
    acc_clu_ema = cluster_acc(y_true=all_label_target_clu, y_pred=all_label_pred_ema_mapped)
    stats_dict.update({'acc_clu_ema': acc_clu_ema})
    print('* acc_clu_ema {acc_clu_ema:.4f}'.format(acc_clu_ema=acc_clu_ema))
        
    return stats_dict, all_prj_features_ema


### [deprecated]
def compute_wordnet_tree_asim(all_synsets_target, all_label_pred, all_label_pred_ema, 
                                tree_metric_name='path_similarity', tree_metric_agg=max, use_norm=False):
    tree_sim = []
    tree_sim_ema = []
    for i in range(len(all_synsets_target)):
        ss_t = all_synsets_target[i]
        ss_p = all_label_pred[i]
        ss_p_ema = all_label_pred_ema[i]
        tree_sim.append(tree_metric_agg([getattr(a, tree_metric_name)(b) for a in ss_t for b in ss_p]))
        tree_sim_ema.append(tree_metric_agg([getattr(a, tree_metric_name)(b) for a in ss_t for b in ss_p_ema]))
    ### 0-1 normalization
    normalize_score = lambda s: (s - s.min())/(s.max() - s.min() + 1e-20) if (s>1).any() else s
    if use_norm:
        tree_sim = normalize_score(np.array(tree_sim)).mean()
        tree_sim_ema = normalize_score(np.array(tree_sim_ema)).mean()
    else:
        tree_sim = np.array(tree_sim).mean()
        tree_sim_ema = np.array(tree_sim_ema).mean()
    return tree_sim, tree_sim_ema



### ======================================================================
### SSL clustering
### ======================================================================
from data.vocab import get_classifier
from collections import Counter
from scipy.optimize import linear_sum_assignment as linear_assignment


def agg_by_pred_cluster(args, pred_kmeans, all_topk_voc, voc_size):
    """
    Args:
        pred_kmeans: np.array([N])
        all_topk_voc: np.array([N x K])
        voc_size: int
    Returns:
        all_clu_pred: tensor([C x V])
    """
    print('agg_by_pred_cluster')
    all_clu_pred = []
    n_count = []
    for i in np.unique(pred_kmeans):
        selected = (pred_kmeans==i)
        n_count.append( selected.sum().item() )
        counter_voc_ind, counter_val = np.unique((all_topk_voc[selected]).ravel(), return_counts=True)
        clu_pred = torch.zeros(args.num_voc) # cluster-wise prob
        clu_pred[torch.from_numpy(counter_voc_ind).long()] = torch.from_numpy(counter_val).float()
        all_clu_pred.append(clu_pred)
    all_clu_pred = torch.stack(all_clu_pred, dim=0).cpu()
    n_count = torch.tensor(n_count).cpu()
    
    all_clu_pred = all_clu_pred/(n_count.view(-1, 1) + 1e-20)
    
    print('is mutex assignment::', all_clu_pred.argmax(dim=-1).size(0)==all_clu_pred.argmax(dim=-1).unique().size(0))
    print('assignment collision num::', len(list(filter(lambda x: x>1, Counter(all_clu_pred.argmax(dim=-1).numpy()).values()))))
    return all_clu_pred


def linear_assign(all_clu_pred, pred_kmeans, all_gt_voc=None, return_results=False):
    print('linear_assign')
    cost_mat = all_clu_pred.cpu().numpy()
    print(f'assignment shape={cost_mat.shape}')
    res_ass = linear_assignment(cost_mat.max() - cost_mat)
    label_voc_kmeans = torch.tensor([res_ass[1][x.item()] for x in pred_kmeans])
    if all_gt_voc is not None:
        inst_acc = (label_voc_kmeans==all_gt_voc).float().mean().item()
        print('instance label acc::', inst_acc)
    else:
        inst_acc = 0
    if return_results:
        return label_voc_kmeans, res_ass, inst_acc
    return label_voc_kmeans, res_ass

def reassign_by_pred_cluster(label_voc_kmeans, loader_f, model, classifier, device, 
                             preextracted_vfeatures=None):
    """ given vocab label set @label_voc_kmeans, 
    Args:
        label_voc_kmeans: cluster-assigned label on vocab
        ...
        preextracted_vfeatures: np.array([N x D])
    Returns:
        cluster_ind: tensor([N]): re-ordered cluster assignment
        cluster_ind_voc: tensor([N]): cluster assignment indiced by vocab
    """
    print('reassign_by_pred_cluster')
    amp_autocast = torch.cuda.amp.autocast
    label_voc_kmeans = label_voc_kmeans.to(device).unique()
    cluster_ind = []
    with tqdm(total=len(loader_f)) as pbar:
        module_eval(model)
        if preextracted_vfeatures is not None:
            N = len(loader_f.dataset)
            batch_size = min(10000, N)
            indices = np.array_split(np.arange(N), N//batch_size)
            with torch.no_grad():
                for group in indices:
                    logits = torch.from_numpy(preextracted_vfeatures[group]).float()
                    logits = logits/logits.norm(dim=-1, keepdim=True)
                    similarity = 100 * logits@classifier.t().cpu()
                    prob = similarity.softmax(-1)
                    cluster_ind.append(prob[:, label_voc_kmeans.cpu()].argmax(dim=-1))
        else:
            for idx_batch, batch in enumerate(loader_f):
                images, label_voc, label_clu, idx_img = batch[:4]
                images = images.to(device)
                with amp_autocast():
                    with torch.no_grad():
                        if preextracted_vfeatures is not None:
                            logits = torch.from_numpy(preextracted_vfeatures[idx_img.cpu().numpy()]).float().to(device)
                        else:
                            logits = model.module.extract_vfeatures(images) if hasattr(model, 'module') else model.extract_vfeatures(images)
                        logits = logits/logits.norm(dim=-1, keepdim=True)
                        similarity = 100 * logits @ classifier.t()
                        prob = similarity.softmax(-1)
                        cluster_ind.append(prob[:, label_voc_kmeans].cpu().argmax(dim=-1))
                pbar.update(1)
    cluster_ind = torch.cat(cluster_ind, dim=0)
    cluster_ind_voc = label_voc_kmeans[cluster_ind]
    mapping_ind = dict(zip(cluster_ind.unique().numpy(), torch.arange(cluster_ind.unique().size(0)).numpy()))
    cluster_ind = torch.tensor([mapping_ind[x.item()] for x in cluster_ind])
    return cluster_ind, cluster_ind_voc


@torch.no_grad()
def compute_ssl_clustering_simple(args, model, model_ssl, loader_f, epoch, 
                           log_writer=None, pred_kmeans_t=None, 
                           return_details=False, load_chatgpt=False, 
                           save_chatgpt_details=False, **kwargs):
    """ SCD
    Returns:
        pred_kmeans_t: tensor([N]): re-ordered cluster assignment
        cluster_ind_voc: tensor([N]): cluster assignment indiced by vocab
    """
    # a = time.time()
    classifier = get_classifier(args)
    classifier = classifier/classifier.norm(dim=-1, keepdim=True)
    amp_autocast = torch.cuda.amp.autocast
    ### collect variables
    prob_k = 1
    all_topk_voc = []
    all_gt_voc = []
    all_label_clu = []
    all_vfeatures = []
    all_soft_pl_ind = []
    all_soft_pl_prob = []
    with tqdm(total=len(loader_f)) as pbar:
        module_eval(model)
        for idx_batch, batch in enumerate(loader_f):
            images, label_voc, label_clu, idx_img = batch[:4]
            images = images.to(args.device)
            with amp_autocast():
                with torch.no_grad():
                    logits = model.module.extract_vfeatures(images) if hasattr(model, 'module') else model.extract_vfeatures(images)
                    logits = logits/logits.norm(dim=-1, keepdim=True)
                    similarity = 100 * logits @ classifier.t()
                    prob = similarity.softmax(-1)
                    prob_topk_ind = prob.topk(k=prob_k, dim=-1).indices
                    all_topk_voc.append(deepcopy(prob_topk_ind.cpu().numpy()))
                    all_gt_voc.append(deepcopy(label_voc))
                    all_label_clu.append(deepcopy(label_clu))
                    all_vfeatures.append(deepcopy(logits.cpu().numpy()))
            pbar.update(1)
    all_topk_voc = np.concatenate(all_topk_voc)
    all_gt_voc = torch.cat(all_gt_voc, dim=0)
    all_label_clu = torch.cat(all_label_clu, dim=0)
    all_vfeatures = np.concatenate(all_vfeatures)

    pred_kmeans_t = pred_kmeans_t.numpy() if isinstance(pred_kmeans_t, torch.Tensor) else pred_kmeans_t
    pred_kmeans_origin = pred_kmeans_t
    for t in range(args.n_iter_cluster_vote):
        record_pred_kmeans_t = pred_kmeans_t
        all_clu_pred = agg_by_pred_cluster(args, pred_kmeans_t, all_topk_voc, voc_size=classifier.size(0))
        label_voc_kmeans, res_ass, inst_acc = linear_assign(all_clu_pred, pred_kmeans_t, all_gt_voc, return_results=True)
        pred_kmeans_t, cluster_ind_voc = reassign_by_pred_cluster(
            label_voc_kmeans, loader_f, model, classifier, args.device, preextracted_vfeatures=all_vfeatures,
            )
        
        set_pred = set(cluster_ind_voc.unique().cpu().numpy().tolist())
        set_gt = set(all_gt_voc.unique().cpu().numpy().tolist())
        n_missing_label = len(set_gt - set_pred)
        n_inter = all_gt_voc[cluster_ind_voc.cpu()==all_gt_voc].unique().shape[0]
        n_union = torch.cat([cluster_ind_voc.cpu(), all_gt_voc]).unique().shape[0]
        iou_voc = n_inter/n_union
        n_missing_label_new = all_gt_voc.unique().shape[0] - n_inter
        print('iou voc::', iou_voc)
        print('missing label::', n_missing_label)
        print('missing label new::', n_missing_label_new)
        acc_cluster = cluster_acc(y_true=all_label_clu.numpy(), y_pred=pred_kmeans_t.numpy())
        print('cluster acc', acc_cluster)
        acc_cluster_origin = cluster_acc(y_true=all_label_clu.numpy(), y_pred=pred_kmeans_origin)
    
    if log_writer is not None:
        log_writer.update(acc_cluster=acc_cluster, head="ssl")
        log_writer.update(inst_acc=inst_acc, head="ssl")
        log_writer.update(n_missing_label=n_missing_label, head="ssl")
        log_writer.update(n_missing_label_new=n_missing_label_new, head="ssl")
        log_writer.update(iou_voc=iou_voc, head='ssl')
        log_writer.update(acc_cluster_origin=acc_cluster_origin, head='ssl')
        

    ### update class-wise agg results (synchronization)
    all_clu_pred = agg_by_pred_cluster(args, pred_kmeans_t.numpy(), all_topk_voc, voc_size=classifier.size(0))
    
    details = {
        'all_clu_pred': all_clu_pred,
        'label_voc_kmeans': label_voc_kmeans,
        'pred_kmeans_t': pred_kmeans_t,
        'cluster_ind_voc': cluster_ind_voc,
        'record_pred_kmeans_t': record_pred_kmeans_t,
        'all_gt_voc': all_gt_voc,
        'all_label_clu': all_label_clu,
        'all_topk_voc': all_topk_voc,
        'classifier': classifier.cpu().clone(),
        'all_vfeatures': all_vfeatures,
    }
    if return_details:
        return pred_kmeans_t, cluster_ind_voc, details
    return pred_kmeans_t, cluster_ind_voc


@torch.no_grad()
def compute_ssl_clustering_simple_label(args, model, model_ssl, loader_f, epoch, 
                           log_writer=None, pred_kmeans_t=None, 
                           return_details=False, load_chatgpt=False, 
                           save_chatgpt_details=False, **kwargs):
    """ SCD
    Returns:
        pred_kmeans_t: tensor([N]): re-ordered cluster assignment
        cluster_ind_voc: tensor([N]): cluster assignment indiced by vocab
    """
    
    classifier = get_classifier(args)
    classifier = classifier/classifier.norm(dim=-1, keepdim=True)
    amp_autocast = torch.cuda.amp.autocast
    ### collect variables
    prob_k = 1
    all_topk_voc = []
    all_gt_voc = []
    all_label_clu = []
    all_vfeatures = []
    all_soft_pl_ind = []
    all_soft_pl_prob = []
    with tqdm(total=len(loader_f)) as pbar:
        module_eval(model)
        for idx_batch, batch in enumerate(loader_f):
            images, label_voc, label_clu, idx_img = batch[:4]
            images = images.to(args.device)
            with amp_autocast():
                with torch.no_grad():
                    logits = model.module.extract_vfeatures(images) if hasattr(model, 'module') else model.extract_vfeatures(images)
                    logits = logits/logits.norm(dim=-1, keepdim=True)
                    similarity = 100 * logits @ classifier.t()
                    prob = similarity.softmax(-1)
                    prob_topk_ind = prob.topk(k=prob_k, dim=-1).indices
                    all_topk_voc.append(deepcopy(prob_topk_ind.cpu().numpy()))
                    all_label_clu.append(deepcopy(label_clu))
                    all_vfeatures.append(deepcopy(logits.cpu().numpy()))
            pbar.update(1)
    all_topk_voc = np.concatenate(all_topk_voc)
    all_label_clu = torch.cat(all_label_clu, dim=0)
    all_vfeatures = np.concatenate(all_vfeatures)

    pred_kmeans_t = pred_kmeans_t.numpy() if isinstance(pred_kmeans_t, torch.Tensor) else pred_kmeans_t
    pred_kmeans_origin = pred_kmeans_t
    for t in range(args.n_iter_cluster_vote):
        record_pred_kmeans_t = pred_kmeans_t
        all_clu_pred = agg_by_pred_cluster(args, pred_kmeans_t, all_topk_voc, voc_size=classifier.size(0))
        label_voc_kmeans, res_ass, inst_acc = linear_assign(all_clu_pred, pred_kmeans_t, None, return_results=True)
        pred_kmeans_t, cluster_ind_voc = reassign_by_pred_cluster(
            label_voc_kmeans, loader_f, model, classifier, args.device, preextracted_vfeatures=all_vfeatures,
            )
        
        acc_cluster = cluster_acc(y_true=all_label_clu.numpy(), y_pred=pred_kmeans_t.numpy())
        print('cluster acc', acc_cluster)
        acc_cluster_origin = cluster_acc(y_true=all_label_clu.numpy(), y_pred=pred_kmeans_origin)
    
    if log_writer is not None:
        log_writer.update(acc_cluster=acc_cluster, head="ssl")
        log_writer.update(inst_acc=inst_acc, head="ssl")
        log_writer.update(acc_cluster_origin=acc_cluster_origin, head='ssl')

    ### update class-wise agg results (synchronization)
    all_clu_pred = agg_by_pred_cluster(args, pred_kmeans_t.numpy(), all_topk_voc, voc_size=classifier.size(0))
    
    details = {
        'all_clu_pred': all_clu_pred,
        'label_voc_kmeans': label_voc_kmeans,
        'pred_kmeans_t': pred_kmeans_t,
        'cluster_ind_voc': cluster_ind_voc,
        'record_pred_kmeans_t': record_pred_kmeans_t,
        'all_gt_voc': all_gt_voc,
        'all_label_clu': all_label_clu,
        'all_topk_voc': all_topk_voc,
        'classifier': classifier.cpu().clone(),
        'all_vfeatures': all_vfeatures,
    }
        
    if return_details:
        return pred_kmeans_t, cluster_ind_voc, details
    return pred_kmeans_t, cluster_ind_voc



### ======================================================================
### LLM - CHATGPT hint
### ======================================================================
from typing import Union, List
import openai
from nltk.corpus import wordnet as wn
# from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

# _tokenizer = _Tokenizer()

# mapping_ids_synset = lambda x: wn.synset_from_pos_and_offset('n', int(x[1:]))
# tree_distance = lambda x, y, z: getattr(x, z)(y)
# mapping_vocidx_to_synsets = lambda anchor, vocab: [
#     mapping_ids_synset(vocab.mapping_global_idx_ids[t]) 
#     for t in vocab.mapping_idx_global_idx[anchor]
#     ]



# def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> torch.LongTensor:
#     if isinstance(texts, str):
#         texts = [texts]

#     sot_token = _tokenizer.encoder["<|startoftext|>"]
#     eot_token = _tokenizer.encoder["<|endoftext|>"]
#     all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
#     result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

#     for i, tokens in enumerate(all_tokens):
#         if len(tokens) > context_length:
#             if truncate:
#                 tokens = tokens[:context_length]
#                 tokens[-1] = eot_token
#             else:
#                 raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
#         result[i, :len(tokens)] = torch.tensor(tokens)

#     return result


# @torch.no_grad()
# def build_classifier_chatgpt(args, all_row_chatgpt_names, model, all_row_key_name=None, tokenizer=None):
#     """ build classifier for chatgpt
#     Args:
#         all_row_chatgpt_names: [[names]]
#     """
#     if all_row_key_name is None: ### single name
#         with open('/home/sheng/sssa/templates_small.json', 'rb') as f: ### template 1
#             templates = json.load(f)['imagenet']
#     else:
#         with open('/home/sheng/sssa/templates_small.json', 'rb') as f: ### template 2
#             templates = json.load(f)[f'{args.dataset}-parent-3']
#     # import pdb; pdb.set_trace()
#     len_t = len(templates)
#     row_classifier = []
#     with tqdm(total=len(all_row_chatgpt_names)) as pbar:
#         for idx, row in enumerate(all_row_chatgpt_names):
#             len_row = len(row)
#             if all_row_key_name is None:
#                 row_t = [ t.format(name) for name in row for t in templates ]
#             else:
#                 row_t = [ t.format(pname, name) for pname, name in zip(all_row_key_name[idx], row) for t in templates ]
#             row_t = tokenize(row_t).to(args.device)
#             if tokenizer is None:
#                 features = model.encode_text(row_t)
#             else:
#                 features = tokenizer(row_t)
#             features = features.view(len_row, len_t, -1).float()
#             features = features/features.norm(dim=-1, keepdim=True)
#             features = features.mean(dim=1)
#             features = features/features.norm(dim=-1, keepdim=True)
#             row_classifier.append(features.cpu())
            
#             pbar.update(1)
#     return row_classifier
    


def load_chatgpt_precomputed_results(args):
    if args.clip_model == 'ViT-B/16':
        if args.uk:
            data = torch.load(f'{PROJECT_HOME}/ipynb/cache/training/cvpr_result-data={args.dataset}-uk={str(args.estimate_k)}-clip.pth') 
        else:
            if len(args.suffix):
                data = torch.load(f'{PROJECT_HOME}/ipynb/cache/training/cvpr_result-data={args.dataset}-clip-suffix={args.suffix}.pth')
            else:
                data = torch.load(f'{PROJECT_HOME}/ipynb/cache/training/cvpr_result-data={args.dataset}-clip.pth')
    elif args.clip_model == 'ViT-L/14':
        data = torch.load(f'{PROJECT_HOME}/ipynb/cache/training/cvpr_result-data={args.dataset}-clip-L.pth')
    else:
        raise NotImplementedError()
    current_epoch_clustering = data['r_pred_kmeans_t'].to(args.device)
    cluster_ind_voc = data['r_cluster_ind_voc'].to(args.device)
    result = {
        'current_epoch_clustering': current_epoch_clustering, 
        'cluster_ind_voc': cluster_ind_voc,
    }
    return result



### ======================================================================
### UTILS
### ======================================================================
import math

def linear_warmup(args, min_val, max_val, curr_iter, max_iter):
    if args.use_warmup_clu:
        return min_val + (max_val - min_val)*min(curr_iter / max_iter, 1)
    else:
        return max_val
    
    
def module_to_device(args, module):
    if args.devices:
        module = nn.DataParallel(module, args.devices)
    else:
        module.to(args.device)
    return module

def module_train(module):
    if hasattr(module, 'module'):
        module.module.train()
    else:
        module.train()
    return module
    
def module_eval(module):
    if hasattr(module, 'module'):
        module.module.train()
    else:
        module.eval()
    return module


def weight_mixup(x, y, min_val, pos_ratio):
    """ linear decay """
    w = min_val + pos_ratio/((1-min_val) - min_val)
    x = x * (1 - w)
    y = y * w
    return x, y

def weight_warmup(w, min_val, pos_ratio):
    """ linear decay """
    w = min_val + max(1, pos_ratio)*(w - min_val)
    return w

def all_to_device(device, *args):
    new_args = []
    for v in args:
        v = v.to(device)
        new_args.append(v)
    return new_args
    