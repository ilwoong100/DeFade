# RecBole-DA

**RecBole-DA** is a library built upon [PyTorch](https://pytorch.org) and [RecBole](https://github.com/RUCAIBox/RecBole) for reproducing and developing data augmentation for sequential recommendation. 

## 1）Highlights

* **Easy-to-use API**:
    Our library provides extensive API based on common data augmentation strategies, users can further develop own new models based on our library.
* **Full Coverage of Classic Methods**:
    We provide seven data augmentation methods based on recommender systems in three major categories.

## 2）Implemented Models

Our library includes algorithms covering three major categories:

* SASRec, DuoRec, BSARec, FEARec, SLIME4Rec, DeFade(fmlphp)

## 3）Requirements

```
recbole>=1.0.0
pytorch>=1.7.0
python>=3.7.0
```

## 4）Quick-Start

With the source code, you can use the provided script for initial usage of our library:

```bash
 python run_seq.py --dataset='beauty' --train_batch_size=256  --model='FMLPHP' --eval_epoch=-1 --shuffle=True --aug=aug --gpu_id=1  --kernel_size=5 --hidden_dropout_prob=0.4 --contrast=None --freq_dropout_prob=0.4 --high_freq_dropout_prob=0.6
```

RecBole-DA is developed and maintained by members from [RUCAIBox](http://aibox.ruc.edu.cn/), the developer is Shuqing Bian ([@fancybian](https://github.com/fancybian)).

## 6） Acknowledgement
This is repo is developed based on [RecBole-DA](https://github.com/RUCAIBox/RecBole-DA)
