import numpy as np
import torch.nn as nn
import torch
import math
import torch.nn.functional as F

def get_positive_expectation(p_samples, measure, average=True):
    """rank_computes the positive part of a divergence / difference.
    Args:
        p_samples: Positive samples.
        measure: Measure to rank_compute for.
        average: Average the result over samples.
    Returns:
        torch.Tensor
    """
    log_2 = math.log(2.)

    if measure == 'GAN':
        Ep = - F.softplus(-p_samples)
    elif measure == 'JSD':
        Ep = log_2 - F.softplus(- p_samples)
    elif measure == 'X2':
        Ep = p_samples ** 2
    elif measure == 'KL':
        Ep = p_samples + 1.
    elif measure == 'RKL':
        Ep = -torch.exp(-p_samples)
    elif measure == 'DV':
        Ep = p_samples
    elif measure == 'H2':
        Ep = 1. - torch.exp(-p_samples)
    elif measure == 'W1':
        Ep = p_samples
    else:
        raise_measure_error(measure)

    if average:
        return Ep.mean()
    else:
        return Ep


def get_negative_expectation(q_samples, measure, average=True):
    """rank_computes the negative part of a divergence / difference.
    Args:
        q_samples: Negative samples.
        measure: Measure to rank_compute for.
        average: Average the result over samples.
    Returns:
        torch.Tensor
    """
    log_2 = math.log(2.)

    if measure == 'GAN':
        Eq = F.softplus(-q_samples) + q_samples
    elif measure == 'JSD':
        Eq = F.softplus(-q_samples) + q_samples - log_2
    elif measure == 'X2':
        Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
    elif measure == 'KL':
        Eq = torch.exp(q_samples)
    elif measure == 'RKL':
        Eq = q_samples - 1.
    elif measure == 'DV':
        Eq = log_sum_exp(q_samples, 0) - math.log(q_samples.size(0))
    elif measure == 'H2':
        Eq = torch.exp(q_samples) - 1.
    elif measure == 'W1':
        Eq = q_samples
    else:
        raise_measure_error(measure)

    if average:
        return Eq.mean()
    else:
        return Eq


def raise_measure_error(measure):
    supported_measures = ['GAN', 'JSD', 'X2', 'KL', 'RKL', 'DV', 'H2', 'W1']
    raise NotImplementedError(
        'Measure `{}` not supported. Supported: {}'.format(measure,
                                                           supported_measures))

def log_sum_exp(x, axis=None):
    """Log sum exp function
    Args:
        x: Input.
        axis: Axis over which to perform sum.
    Returns:
        torch.Tensor: log sum exp
    """
    x_max = torch.max(x, axis)[0]
    y = torch.log((torch.exp(x - x_max)).sum(axis)) + x_max
    return y



class Aggregator(nn.Module):
    def __init__(self, batch_size, input_dim, output_dim, act, self_included):
        super(Aggregator, self).__init__()
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act = act
        self.self_included = self_included

    def forward(self, self_vectors, neighbor_vectors, masks):
        # self_vectors: [batch_size, -1, input_dim]
        # neighbor_vectors: [batch_size, -1, 2, n_neighbor, input_dim]
        # masks: [batch_size, -1, 2, n_neighbor, 1]
        entity_vectors = torch.mean(neighbor_vectors * masks, dim=-2)  # [batch_size, -1, 2, input_dim]
        outputs = self._call(self_vectors, entity_vectors)
        return outputs


class ConcatAggregator(Aggregator):
    def __init__(self, batch_size, input_dim, output_dim, act=lambda x: x, self_included=True):
        super(ConcatAggregator, self).__init__(batch_size, input_dim, output_dim, act, self_included)

        multiplier = 3 if self_included else 2

        self.layer = nn.Linear(self.input_dim * multiplier, self.output_dim)
        nn.init.xavier_uniform_(self.layer.weight)

    def _call(self, self_vectors, entity_vectors):
        # self_vectors: [batch_size, -1, input_dim]
        # entity_vectors: [batch_size, -1, 2, input_dim]

        output = entity_vectors.view([-1, self.input_dim * 2])  # [-1, input_dim * 2]
        if self.self_included:
            self_vectors = self_vectors.view([-1, self.input_dim])  # [-1, input_dim]
            output = torch.cat([self_vectors, output], dim=-1)  # [-1, input_dim * 3]
        output = self.layer(output)  # [-1, output_dim]
        output = output.view([self.batch_size, -1, self.output_dim])  # [batch_size, -1, output_dim]

        return self.act(output)


def evaluate(entity_pairs, labels, args, model):
    acc_list = []
    scores_list = []

    s = 0
    while s + args.batch_size <= len(labels):
        acc, scores = model.test_step(model, get_feed_dict(
            entity_pairs, None, labels, s, s + args.batch_size))
        acc_list.extend(acc)
        scores_list.extend(scores)
        s += args.batch_size

    return float(np.mean(acc_list)), np.array(scores_list)


def rank_compute(x):
    return round(float(0.5*np.mean(x)), 4)


def calculate_ranking_metrics(triplets, scores, true_relations):
    for i in range(scores.shape[0]):
        head, tail, relation = triplets[i]
        for j in true_relations[head, tail] - {relation}:
            scores[i, j] -= 1.0

    sorted_indices = np.argsort(-scores, axis=1)
    relations = np.array(triplets)[0:scores.shape[0], 2]
    sorted_indices -= np.expand_dims(relations, 1)
    zero_coordinates = np.argwhere(sorted_indices == 0)
    rankings = zero_coordinates[:, 1] + 1

    mrr = rank_compute(1 / rankings)
    hit1 = rank_compute(rankings <= 1)
    hit3 = rank_compute(rankings <= 3)
    hit10 = rank_compute(rankings <= 10)
    return mrr, hit1, hit3, hit10


def get_feed_dict(entity_pairs, train_edges, labels, start, end):
    feed_dict = {}
    feed_dict["entity_pairs"] = entity_pairs[start:end]
    if train_edges is not None:
        feed_dict["train_edges"] = train_edges[start:end]
    else:
        # for evaluation no edges should be masked out
        feed_dict["train_edges"] = torch.LongTensor(np.array([-1] * (end - start), np.int32)).cuda() if args.cuda \
            else torch.LongTensor(np.array([-1] * (end - start), np.int32))

    feed_dict["labels"] = labels[start:end]

    return feed_dict