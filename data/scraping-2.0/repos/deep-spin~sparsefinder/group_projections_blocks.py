import math

import torch
import numpy as np
from torch import Tensor
from entmax import entmax15

from utils import get_length_mask, dot_product_and_mask, subsequent_mask, neighbours_mask


def quantize(
    x: Tensor,
    bucket_size: int,
    lengths: Tensor,
    enforce_equal_size: bool = False,
    return_lower_bounds: bool = False,
    temperature: float = 1.
):
    """
    Args:
        x: projected query or key (batch, heads, n, projections)
        bucket_size: number of tokens per bucket
        lengths: tensor (batch,) - necessary for masking padding
        enforce_equal_size: makes all buckets with the same number of tokens.
            If not, buckets will have tokens falling within the same interval
        return_lower_bounds: return also a tensor with the lowest value in each
            bucket
        temperature: coefficient to multiply projected values before tanh
            (only used with variable-sized buckets)

    Return:
        buckets (batch, heads, n, projections): the bucket each token has
            been assigned to at each projection
        lower_bounds [Optional] (num_intervals, rounds): the lowest value
            assigned to each bucket
    """
    batch_size, num_heads, n, num_projections = x.shape

    mask = get_length_mask(lengths, n)
    x = x.masked_fill(~mask.view(batch_size, 1, n, 1), np.inf)
    num_intervals = math.ceil(n / bucket_size)

    if enforce_equal_size:
        modulo = n % bucket_size
        if modulo != 0:
            # pad
            shape = [batch_size, num_heads, bucket_size - modulo, num_projections]
            padding = torch.full(shape, np.inf).to(x.device)
            x = torch.cat([x, padding], dim=2)

        # each round will have all tokens sorted by value
        inds = x.argsort(2)
        rev_inds = inds.argsort(2)
        buckets = rev_inds // bucket_size

        # remove the padding needed for having a multiple of `intervals`
        buckets = buckets[:, :, :n]

    else:
        step = 2 / num_intervals
        num_boundaries = num_intervals - 1
        boundaries = torch.linspace(-1 + step, 1 - step, num_boundaries)
        boundaries = boundaries.to(x.device)
        # torch.bucketize fails on tensors requiring gradient
        x = torch.tanh(temperature * x.detach())
        buckets = torch.bucketize(x, boundaries)

    if return_lower_bounds:
        raise NotImplementedError

    return buckets


def group_qk(
    queries: torch.Tensor,
    keys: torch.Tensor,
    lengths: torch.LongTensor,
    strategy: str = 'distance_l2',
    add_causal_mask: bool = False,
    window_size: int = 0,
    add_cls_mask: bool = False,
    add_last_mask: bool = False,
    return_buckets: bool = False,
    clusters=None,
    **kwargs
):
    """
    Group queries and keys according to a pre-defined `strategy`.

    Args:
        queries: float tensor of shape (bs, h, n, hdim)
        keys: float tensor of shape (bs, h, n, hdim)
        lengths: long tensor of shape (bs,)
        strategy: str in [
            'distance_cos', 'distance_euclidean', 'distance_l2', 'distance_entmax', 'bucketing_fixed',
            'bucketing_dynamic', 'clustering_kmeans', 'simulated_routing', 'simulated_routing_extended',
            'simulated_bigbird', 'simulated_longformer', 'simulated_openai_sparse', 'simulated_reformer']
        window_size: length of diagonal pattern
        add_causal_mask: whether the attention is causal (aka decoder attention)
        add_cls_mask: whether to add connections to the CLS (first) token
        add_last_mask: whether to add connections to the last token
        return_buckets: whether to return clusters/buckets
        clusters: A cluster manager object (see clustering_blocks.py)
        kwargs: passed to the function that implements the specified strategy

    Return:
        pairwise_mask: torch.BoolTensor of shape (bs, h, n, n), with 1s indicating
            connected edges (grouped q and k) and 0s disconnected edges.
        Optional (if `return_buckets` is True): returns buckets/clusters
            buckets_q: torch.LongTensor (bs, h, n, num_buckets)
            buckets_k: torch.LongTensor (bs, h, n, num_buckets)
    """
    buckets_q = buckets_k = None

    if 'distance_' in strategy:
        distance = strategy.split('_')[1]
        pairwise_conns = group_by_distance(queries, keys, lengths, dist=distance, **kwargs)

    elif 'bucketing_' in strategy:
        pairwise_conns, buckets_q, buckets_k = group_by_buckets(queries, keys, lengths, **kwargs)

    elif 'clustering_' in strategy:
        pairwise_conns, buckets_q, buckets_k = group_by_clusters(queries, keys, clusters, **kwargs)

    elif 'simulated_' in strategy:
        arch = strategy.split('_')[1]
        pairwise_conns = group_by_simulated_transformers(queries, keys, lengths, arch=arch, clusters=clusters, **kwargs)

    else:
        raise NotImplementedError

    batch_size, num_heads, seq_len, _ = queries.shape
    device = queries.device

    # create pairwise_mask for padding
    pad_mask = get_length_mask(lengths, seq_len)
    pairwise_mask = pad_mask.unsqueeze(-1) & pad_mask.unsqueeze(1)

    # consider causal attention for padding
    if add_causal_mask:
        causal_mask = subsequent_mask(seq_len).bool().to(device)
        pairwise_mask = pad_mask.unsqueeze(-1) & pad_mask.unsqueeze(1)
        pairwise_mask = pairwise_mask & causal_mask.unsqueeze(0)

    # add window, cls, and last token mask:
    if window_size is not None and window_size > 0:
        win_mask = neighbours_mask(seq_len, window_size).unsqueeze(0).unsqueeze(1).bool().to(device)
        pairwise_conns |= win_mask
    if add_cls_mask is True:
        pairwise_conns[:, :, 0, :] = True  # from cls
        pairwise_conns[:, :, :, 0] = True  # to cls
    if add_last_mask is True:
        pairwise_conns[:, :, -1, :] = True  # from last
        pairwise_conns[:, :, :, -1] = True  # to last

    # fix some corners cases
    if pairwise_conns.shape[0] == 1:
        pairwise_conns = pairwise_conns.repeat(batch_size, 1, 1, 1)
    if pairwise_conns.shape[1] == 1:
        pairwise_conns = pairwise_conns.repeat(1, num_heads, 1, 1)

    # ignore padding:
    pairwise_conns = pairwise_conns.masked_fill(~pairwise_mask.unsqueeze(1), False)

    if return_buckets:
        return pairwise_conns, buckets_q, buckets_k
    return pairwise_conns


def group_by_buckets(qproj, kproj, lengths, bucket_size=4, enforce_same_size=True, temperature=1., **kwargs):
    # bucketize
    buckets_q = quantize(qproj, bucket_size, lengths, enforce_same_size, temperature=temperature)
    buckets_k = quantize(kproj, bucket_size, lengths, enforce_same_size, temperature=temperature)
    # cross all Q and K to find out when they match
    # shape is (batch, heads, query, key, projection)
    pairwise_conns = buckets_q.unsqueeze(3) == buckets_k.unsqueeze(2)
    # consider matches across any projection
    # shape is (batch, heads, query, key)
    pairwise_conns = pairwise_conns.sum(4) > 0
    return pairwise_conns, buckets_q, buckets_k


def group_by_clusters(qproj, kproj, clusters=None, top_clusters=1, **kwargs):
    # clusterize
    clusters_q = clusters.predict(qproj, top_clusters=top_clusters)
    clusters_k = clusters.predict(kproj, top_clusters=top_clusters)
    # cross all Q and K to find out when they match
    # shape is (batch, heads, query, key, top_clusters)
    pairwise_conns = clusters_q.unsqueeze(3) == clusters_k.unsqueeze(2)
    # consider matches across any top cluster
    # shape is (batch, heads, query, key)
    pairwise_conns = pairwise_conns.sum(4) > 0
    return pairwise_conns, clusters_q, clusters_k


def group_by_distance(qproj, kproj, lengths, dist='cos', threshold=0.5, **kwargs):
    if dist == 'cos':
        qproj_norm = qproj.norm(p=2, dim=-1).unsqueeze(-1)
        kproj_norm = kproj.norm(p=2, dim=-1).unsqueeze(-1)
        cos_sim = (qproj / qproj_norm) @ (kproj / kproj_norm).transpose(-1, -2)
        pairwise_ds = 1 - cos_sim
    elif dist == 'euclidean':
        pairwise_ds = torch.cdist(qproj, kproj, p=2.0)
    elif dist == 'l2':
        pairwise_ds = torch.cdist(qproj, kproj, p=2.0) ** 2
    elif dist == 'entmax':
        dots = dot_product_and_mask(qproj, kproj, lengths)
        att_dist = entmax15(dots, dim=-1)
        pairwise_ds = (att_dist == 0).float() * 999.0 + (att_dist > 0).float() * 0.0
    else:  # random
        bs, nh, slen, _ = qproj.shape
        pairwise_ds = torch.rand(bs, nh, slen, slen, device=qproj.device)
    pairwise_ds_thresholded = pairwise_ds < threshold
    return pairwise_ds_thresholded


def group_by_simulated_transformers(qproj, kproj, lengths, arch='longformer', clusters=None, **kwargs):
    batch_size, num_heads, seq_len, hidden_size = qproj.shape
    device = qproj.device
    mask = get_length_mask(lengths, seq_len)

    if arch == 'bigbird':
        from bigbird import bigbird_simulated_attention
        pairwise_mask = mask.unsqueeze(-1) & mask.unsqueeze(1)
        pairwise_mask.unsqueeze_(1)
        x = bigbird_simulated_attention(
            pairwise_mask.cpu().numpy(),
            num_attention_heads=num_heads,
            num_rand_blocks=kwargs['bigbird_num_random_blocks'],  # default from bigbird repo is 3
            from_seq_length=seq_len,
            to_seq_length=seq_len,
            from_block_size=kwargs['bigbird_block_size'],
            to_block_size=kwargs['bigbird_block_size'],
            max_seq_len=seq_len
        )
        pairwise_conns = x.to(device).bool()
    elif arch == 'longformer':
        from longformer import longformer_simulated_attention
        sim_mask = longformer_simulated_attention(
            mask,
            window_size=1,
            dilation=0,
            max_globals_per_sample=kwargs['longformer_max_globals'],
            min_globals_per_sample=2,
        )
        pairwise_conns = sim_mask.bool().unsqueeze(1)
    elif arch == 'openai_sparse':
        from openai_sparse import openai_sparse_simulated_attention
        sim_mask = openai_sparse_simulated_attention(
            mask,
            hidden_size,
            num_heads,
            stride=kwargs['openai_sparse_stride'],
            expressivity=2,
        )
        pairwise_conns = sim_mask.bool().unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1, 1)
    elif arch in ['reformer', 'reformer_rounds']:
        from reformer import reformer_simulated_attention
        # https://github.com/lucidrains/reformer-pytorch/blob/master/reformer_pytorch/reformer_pytorch.py#L522
        num_hashes = hidden_size if arch == 'reformer_rounds' else 1
        qk = torch.cat([qproj, kproj], dim=-1)
        sim_mask = reformer_simulated_attention(
            qk,
            lsh_attn_chunk_length=None,  # default from reformer is 4 or 8 for longer seqs
            num_buckets=kwargs['reformer_num_buckets'],
            num_hashes=num_hashes,
            mask=mask
        )
        pairwise_conns = sim_mask.bool()
    elif arch in ['routing', 'routing_trained', 'routing_extended']:
        sim_mask = clusters.predict(qproj, kproj)
        pairwise_conns = sim_mask.bool().to(device)
    else:
        raise NotImplementedError

    return pairwise_conns
