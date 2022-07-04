# InducT-GCN: Inductive Graph Convolutional Networks for Text Classification
This repository contains code for paper [InducT-GCN: Inductive Graph Convolutional Networks for Text Classification](https://arxiv.org/abs/2206.00265)

<h3 align="center">
  <b>Wang, K., Han, S. C., & Poon, J. (2020) <br/><a href="https://arxiv.org/abs/2206.00265">InducT-GCN: Inductive Graph Convolutional Networks for Text Classification]</a><br/>In ICPR 2022</b></span>
</h3>

## How to Use
### Reproducing results
Simply run `python main.py --dataset 'R8' --train_size 0.05`
### Arguments description
| Argument     | Default   | Description |
| ----------- | ----------- |----------- |
| dataset | R8 | Dataset string: R8, R52, OH, 20NGnew, MR       |
| train_size  | 1 | If it is larger than 1, it means the number of training samples. If it is from 0 to 1, it means the proportion of the original training set.        |
| test_size  | 1 | If it is larger than 1, it means the number of training samples. If it is from 0 to 1, it means the proportion of the original test set.|
| remove_limit  | 2 | Remove the words showing fewer than 2 times |
| use_gpu  | 1 | Whether to use GPU, 1 means True and 0 means False. If True and no GPU available, will use CPU instead. |
| shuffle_seed  | None | If not specified, train/val is shuffled differently in each experiment. |
| hidden_dim  | 200 | The hidden dimension of GCN model |
| dropout  | 0.5 | The dropout rate of GCN model |
| learning_rate  | 0.02 | Learning rate, and the optimizer is Adam |
| weight_decay  | 0 | Weight decay, normally it is 0 |
| early_stopping  | 10 | Number of epochs of early stopping |
| epochs  | 200 | Number of maximum epochs |
| multiple_times  | 10 | Running multiple experiments, each time the train/val split is different |
| easy_copy  | 1 | For easy copy of the experiment results. 1 means True and 0 means False. |

## Citation
If you find this paper useful, please cite it by 
```
@article{wang2022induct,
  title={InducT-GCN: Inductive Graph Convolutional Networks for Text Classification},
  author={Wang, Kunze and Han, Soyeon Caren and Poon, Josiah},
  journal={arXiv preprint arXiv:2206.00265},
  year={2022}
}
```
Since the conference is not held yet, the citation is arXiv version for now.

## Acknowledgement
Part of the code is inspired by https://github.com/tkipf/pygcn and https://github.com/yao8839836/text_gcn, but has been modified.
