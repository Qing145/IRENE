import torch
import numpy as np
from collections import defaultdict
from model import IRENE
from utils import *


args = None


def train(model_args, data):
    global args, model, sess
    args = model_args

    # extract data
    triplets, n_relations, neighbor_params = data

    train_triplets, valid_triplets, test_triplets = triplets
    train_edges = torch.LongTensor(np.array(range(len(train_triplets)), np.int32))
    train_entity_pairs = torch.LongTensor(np.array([[triplet[0], triplet[1]] for triplet in train_triplets], np.int32))
    valid_entity_pairs = torch.LongTensor(np.array([[triplet[0], triplet[1]] for triplet in valid_triplets], np.int32))
    test_entity_pairs = torch.LongTensor(np.array([[triplet[0], triplet[1]] for triplet in test_triplets], np.int32))

    train_labels = torch.LongTensor(np.array([triplet[2] for triplet in train_triplets], np.int32))
    valid_labels = torch.LongTensor(np.array([triplet[2] for triplet in valid_triplets], np.int32))
    test_labels = torch.LongTensor(np.array([triplet[2] for triplet in test_triplets], np.int32))

    # define the model
    model = IRENE(args, n_relations, neighbor_params)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.lr,
        # weight_decay=args.l2,
    )

    if args.cuda:
        model = model.cuda()
        train_labels = train_labels.cuda()
        valid_labels = valid_labels.cuda()
        test_labels = test_labels.cuda()
        if args.use_context:
            train_edges = train_edges.cuda()
            train_entity_pairs = train_entity_pairs.cuda()
            valid_entity_pairs = valid_entity_pairs.cuda()
            test_entity_pairs = test_entity_pairs.cuda()

    # prepare for top-k evaluation
    true_relations = defaultdict(set)
    for head, tail, relation in test_triplets:
        true_relations[(head, tail)].add(relation)
    best_valid_acc = 0.0
    final_res = None  # acc, mrr, mr, hit1, hit3, hit10

    print('start training ...')

    for step in range(args.epoch):

        # shuffle training data
        index = np.arange(len(train_labels))
        np.random.shuffle(index)
        if args.use_context:
            train_entity_pairs = train_entity_pairs[index]
            train_edges = train_edges[index]

        train_labels = train_labels[index]

        # training
        s = 0
        while s + args.batch_size <= len(train_labels):
            loss = model.train_step(model, optimizer, get_feed_dict(
                train_entity_pairs, train_edges, train_labels, s, s + args.batch_size))
            s += args.batch_size

        # evaluation
        print('epoch %2d   ' % step, end='')
        train_acc, _ = evaluate(train_entity_pairs, train_labels, args, model)
        valid_acc, _ = evaluate(valid_entity_pairs, valid_labels, args, model)
        test_acc, test_scores = evaluate(test_entity_pairs, test_labels, args, model)

        # show evaluation result for current epoch
        current_res = 'acc: %.4f' % test_acc
        print('train acc: %.4f   valid acc: %.4f   test acc: %.4f' % (train_acc, valid_acc, test_acc))
        mrr, hit1, hit3, hit10 = calculate_ranking_metrics(test_triplets, test_scores, true_relations)
        current_res += '   mrr: %.4f    h1: %.4f   h3: %.4f   h10: %.4f' % (mrr, hit1, hit3, hit10)
        print('           mrr: %.4f     h1: %.4f   h3: %.4f   h10: %.4f' % (mrr, hit1, hit3, hit10))
        print()

        # update final results according to validation accuracy
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            final_res = current_res

    # show final evaluation result
    print('final results\n%s' % final_res)




