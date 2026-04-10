import torch

def bernoulli_sample(logits, k):
    probs = torch.sigmoid(logits)
    m = torch.distributions.Bernoulli(probs)
    actions = m.sample()
    log_prob = m.log_prob(actions)
    log_prob_total = log_prob.squeeze(-1).sum(-1)
    return actions, log_prob_total


def multinomial_sample(logits, k):
    probs = torch.sigmoid(logits).squeeze(-1)
    samples = torch.multinomial(probs, k, replacement=False)

    # 0/1 mask
    actions = torch.zeros_like(probs)
    actions = actions.scatter(1, samples, 1)

    # log_prob for PG
    log_prob = torch.log(probs + 1e-8)
    log_prob_total = log_prob.gather(1, samples).sum(-1)
    return actions, log_prob_total


def gumbel_top_k(logits, K):
    # logtis: shape [b,n,1]
    probs = torch.softmax(logits.squeeze(-1), dim=1)

    # Gumbel noise
    gumbel = -torch.log(-torch.log(torch.rand_like(probs)))
    scores = torch.log(probs + 1e-8) + gumbel

    # top-k indices
    _, topk_idx = scores.topk(K, dim=-1)

    # 0/1 mask
    actions = torch.zeros_like(probs)
    actions = actions.scatter(1, topk_idx, 1)

    # log_prob for PG
    log_prob = torch.log(probs + 1e-8)
    log_prob_total = log_prob.gather(1, topk_idx).sum(-1)

    return actions, log_prob_total

def split_by_mask(input_ids, mask):
    # 去掉 batch 维
    input_ids = input_ids.squeeze(0)
    mask = mask.squeeze(0)

    # 找 True 的索引
    true_indices = torch.nonzero(mask, as_tuple=False).squeeze(1)

    if true_indices.numel() == 0:
        return []

    # 找连续段的分界点
    # 当相邻索引差 > 1 时说明断开
    splits = torch.where(true_indices[1:] - true_indices[:-1] > 1)[0] + 1

    # 按分界点切分
    segments = torch.split(true_indices, splits.tolist())

    # 用索引取回 input_ids
    result = [input_ids[idx] for idx in segments]

    return result
