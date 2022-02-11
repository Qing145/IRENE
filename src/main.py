import argparse
from data_loader import load_data
from train import train
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def print_setting(args):
    print()
    print('=============================================')
    print('dataset: ' + args.dataset)
    print('epoch: ' + str(args.epoch))
    print('batch_size: ' + str(args.batch_size))
    print('dim: ' + str(args.dim))
    print('l2: ' + str(args.l2))
    print('lr: ' + str(args.lr))
    print('context_hops: ' + str(args.context_hops))
    print('neighbor_agg: ' + args.neighbor_agg)
    print('=============================================')
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=False, help='use gpu', action='store_true')

    # ===== wn18rr ===== #
    parser.add_argument('--dataset', type=str, default='wn18rr', help='dataset name')
    parser.add_argument('--epoch', type=int, default=200, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--dim', type=int, default=100, help='hidden dimension')
    parser.add_argument('--l2', type=float, default=1e-7, help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--feature_type', type=str, default='id', help='type of relation features: id, bow, bert')
    parser.add_argument('--use_context', type=bool, default=True, help='whether use relational context')
    parser.add_argument('--context_hops', type=int, default=3, help='number of context hops')
    parser.add_argument('--neighbor_samples', type=int, default=20, help='number of sampled neighbors for one hop')
    parser.add_argument('--neighbor_agg', type=str, default='GAT', help='neighbor aggregator: GraphSage, GIN, GAT')

    args = parser.parse_args()
    print_setting(args)
    data = load_data(args)
    train(args, data)


if __name__ == '__main__':
    main()
