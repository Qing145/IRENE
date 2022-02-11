import os
import re
import pickle
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer


entity2edge_set = defaultdict(set)  # entity id -> set of (both incoming and outgoing) edges connecting to this entity
entity2edges = []  # each row in entity2edges is the sampled edges connecting to this entity
edge2entities = []  # each row in edge2entities is the two entities connected by this edge
edge2relation = []  # each row in edge2relation is the relation type of this edge

e2re = defaultdict(set)  # entity index -> set of pair (relation, entity) connecting to this entity


def read_entities(file_name):
    d = {}
    file = open(file_name)
    for line in file:
        index, name = line.strip().split('\t')
        d[name] = int(index)
    file.close()

    return d


def read_relations(file_name):
    bow = []
    count_vec = CountVectorizer()

    d = {}
    file = open(file_name)
    for line in file:
        index, name = line.strip().split('\t')
        d[name] = int(index)

        if args.feature_type == 'bow' and not os.path.exists('../data/' + args.dataset + '/bow.npy'):
            tokens = re.findall('[a-z]{2,}', name)
            bow.append(' '.join(tokens))
    file.close()

    if args.feature_type == 'bow' and not os.path.exists('../data/' + args.dataset + '/bow.npy'):
        bow = count_vec.fit_transform(bow)
        np.save('../data/' + args.dataset + '/bow.npy', bow.toarray())

    return d


def read_triplets(file_name):
    data = []

    file = open(file_name)
    for line in file:
        head, relation, tail = line.strip().split('\t')

        head_idx = entity_dict[head]
        relation_idx = relation_dict[relation]
        tail_idx = entity_dict[tail]

        data.append((head_idx, tail_idx, relation_idx))
    file.close()

    return data



def build_kg(train_data):
    for edge_idx, triplet in enumerate(train_data):
        head_idx, tail_idx, relation_idx = triplet

        if args.use_context:
            entity2edge_set[head_idx].add(edge_idx)
            entity2edge_set[tail_idx].add(edge_idx)
            edge2entities.append([head_idx, tail_idx])
            edge2relation.append(relation_idx)
    null_entity = len(entity_dict)
    null_relation = len(relation_dict)
    null_edge = len(edge2entities)
    edge2entities.append([null_entity, null_entity])
    edge2relation.append(null_relation)

    for i in range(len(entity_dict) + 1):
        if i not in entity2edge_set:
            entity2edge_set[i] = {null_edge}
        sampled_neighbors = np.random.choice(list(entity2edge_set[i]), size=args.neighbor_samples,
                                             replace=len(entity2edge_set[i]) < args.neighbor_samples)
        entity2edges.append(sampled_neighbors)



def get_h2t(train_triplets, valid_triplets, test_triplets):
    head2tails = defaultdict(set)
    for head, tail, relation in train_triplets + valid_triplets + test_triplets:
        head2tails[head].add(tail)
    return head2tails


def load_data(model_args):
    global args, entity_dict, relation_dict
    args = model_args
    directory = '../data/' + args.dataset + '/'

    print('reading entity dict and relation dict ...')
    entity_dict = read_entities(directory + 'entities.dict')
    relation_dict = read_relations(directory + 'relations.dict')

    print('reading train, validation, and test data ...')
    train_triplets = read_triplets(directory + 'train.txt')
    valid_triplets = read_triplets(directory + 'valid.txt')
    test_triplets = read_triplets(directory + 'test.txt')

    print('processing the knowledge graph ...')
    build_kg(train_triplets)

    triplets = [train_triplets, valid_triplets, test_triplets]

    neighbor_params = [np.array(entity2edges), np.array(edge2entities), np.array(edge2relation)]


    return triplets, len(relation_dict), neighbor_params