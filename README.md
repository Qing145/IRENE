# Inductive Knowledge Graph Completion with Relation Network

Pytorch-based implementation of IRENE, and the description of the model and the results can be found in the paper: "Infomax RElation NEtwork For Inductive Knowledge Graph Completion" [1].

## Requirements

All the required packages can be installed by running `pip install -r requirements.txt`.

## Inductive Knowledge Graph Completion

To replicate the experiments from our paper [1], run (for WN18RR):

```
python main.py --dataset "wn18rr" --batch_size 256 --dim 100 --lr 0.005
```

Note that IRENE performs an expensive compilation step the first time a computational graph is executed. This can take several minutes to complete.
