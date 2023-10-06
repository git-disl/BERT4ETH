# BERT4ETH

I have reorganized the code and tested it recently. The code should be able to reproduce the results presented in the paper. (2023/08/27)

This is the repo for the code and datasets used in the paper [BERT4ETH: A Pre-trained Transformer for Ethereum Fraud Detection](https://dl.acm.org/doi/abs/10.1145/3543507.3583345), accepted by the ACM Web conference (WWW) 2023.

Here you can find our [slides](https://github.com/git-disl/BERT4ETH/blob/master/Material/BERT4ETH_WWW23_slides.pdf).

## Getting Start

### Requirements
* Python >= 3.6
* TensorFlow >= 1.4.0

I use python 3.9, tensorflow 2.9.2 with CUDA 11.2, numpy 1.19.5.

### Preprocess dataset 

#### Step 1: Download dataset from Google Drive. 
* Transaction Dataset:
* * [Phishing Account](https://drive.google.com/file/d/11UAhLOcffzLyPhdsIqRuFsJNSqNvrNJf/view?usp=sharing)

* * [De-anonymization(ENS)](https://drive.google.com/file/d/1Yveis90jCx-nIA6pUL_4SUezMsVJr8dp/view?usp=sharing)

* * [De-anonymization(Tornado)](https://drive.google.com/file/d/1DMbPSZMSvTYMKUZg3oYKFrjPo2_jeeG4/view?usp=sharing)

* * [Normal Account](https://drive.google.com/file/d/1-htLUymg1UxDrXcI8tslU9wbn0E1vl9_/view?usp=sharing)

* [ERC-20 Log Dataset (all in one)](https://drive.google.com/file/d/1mB2Tf7tMq5ApKKOVdctaTh2UZzzrAVxq/view?usp=sharing)

The master branch hosts the basic BERT4ETH model. If you wish to run the basic BERT4ETH model, there is no need to download the ERC-20 log dataset. Advanced features such as In/out separation and ERC20 log can be found in the old branch.

#### Step 2: Unzip dataset under the directory of "BERT4ETH/Data/" 

```sh
cd BERT4ETH/Data; # Labels are already included
unzip ...;
``` 

#### Step 3: Transaction Sequence Generation

```sh
cd Model;
python gen_seq.py --bizdate=bert4eth_exp
```


### Pre-training

<!-- 
#### Step 0: Model Configuration

The configuration file is "Model/bert_config.json"
```
{
  "attention_probs_dropout_prob": 0.2,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.2,
  "hidden_size": 64,
  "initializer_range": 0.02,
  "intermediate_size": 64,
  "max_position_embeddings": 50,
  "num_attention_heads": 2,
  "num_hidden_layers": 8,
  "type_vocab_size": 2,
  "vocab_size": 3000000
}
```
-->

#### Step 1: Pre-training Data Generation from Sequence 

[//]: # (&#40;Masking, I/O separation and ERC20 log&#41;)

```sh
python gen_pretrain_data.py --bizdate=bert4eth_exp  \ 
                            --max_seq_length=100  \
                            --dupe_factor=10 \
                            --masked_lm_prob=0.8 
```


#### Step 2: Pre-train BERT4ETH 

```sh
python run_pretrain.py --bizdate=bert4eth_exp \
                       --max_seq_length=100 \
                       --epoch=5 \
                       --batch_size=256 \
                       --learning_rate=1e-4 \
                       --num_train_steps=1000000 \
                       --save_checkpoints_steps=8000 \
                       --neg_strategy=zip \
                       --neg_sample_num=5000 \ 
                       --neg_share=True \ 
                       --checkpointDir=bert4eth_exp 
```


| Parameter                | Description                                                                        |
|--------------------------|------------------------------------------------------------------------------------|
| `bizdate`                | The signature for this experiment run.                                             |
| `max_seq_length`         | The maximum length of BERT4ETH.                                                    |
| `masked_lm_prob`         | The probability of masking an address.                                             |
| `epochs`                 | Number of training epochs, default = `5`.                                          |
| `batch_size`             | Batch size, default = `256`.                                                       |
| `learning_rate`          | Learning rate for the optimizer (Adam), default = `1e-4`.                          |
| `num_train_steps`        | The maximum number of training steps, default = `1000000`,                         |
| `save_checkpoints_steps` | The parameter controlling the step of saving checkpoints, default = `8000`.        |
| `neg_strategy`           | Strategy for negative sampling, default `zip`, options (`uniform`, `zip`, `freq`). |
| `neg_share`              | Whether enable in-batch sharing strategy, default = `True`.                        |
| `neg_sample_num`         | The negative sampling number for one batch, default = `5000`.                      |
| `checkpointDir`          | Specify the directory to save the checkpoints.                                     |


#### Step 3: Output Representation

```sh
python output_embed.py --bizdate=bert4eth_exp \
                       --init_checkpoint=bert4eth_exp/model_104000 \
                       --max_seq_length=100 \
                       --neg_sample_num=5000 \
                       --neg_strategy=zip \
                       --neg_share=True
```

I have generated a version of [embedding file](https://drive.google.com/file/d/1mQgO1LalWhjeR064VhPRmkLl_ztSTroW/view?usp=sharing), you can unzip it under the directory of "Model/inter_data/" and test the results.

### Testing on output account representation

#### Phishing Account Detection
```sh
python run_phishing_detection.py --init_checkpoint=bert4eth_exp/model_104000 # Random Forest (RF)

python run_phishing_detection_dnn.py --init_checkpoint=bert4eth_exp/model_104000 # DNN, better than RF
```

#### De-anonymization (ENS dataset)

```sh
python run_dean_ENS.py --metric=euclidean \
                       --init_checkpoint=bert4eth_exp/model_104000
```

<!-- 
#### De-anonymization (Tornado Cash)

```sh

python run_dean_Tornado.py --metric=euclidean \

                           --init_checkpoint=bert4eth_exp/model_104000

```
-->

### Fine-tuning for phishing account detection
```sh

python gen_finetune_phisher_data.py --bizdate=bert4eth_exp \ 
                                    --max_seq_length=100 
```

```sh
python run_finetune_phisher.py --init_checkpoint=bert4eth_exp/model_104000 \
                               --bizdate=bert4eth_exp \ 
                               --max_seq_length=100 \ 
                               --checkpointDir=tmp
```

-----
## Citation

If you find this repository useful, please give us a star and cite our paper : ) Thank you!
```
@inproceedings{hu2023bert4eth,
  title={BERT4ETH: A Pre-trained Transformer for Ethereum Fraud Detection},
  author={Hu, Sihao and Zhang, Zhen and Luo, Bingqiao and Lu, Shengliang and He, Bingsheng and Liu, Ling},
  booktitle={Proceedings of the ACM Web Conference 2023},
  pages={2189--2197},
  year={2023}
}
```

-----
## Q&A

If you have any questions, you can either open an issue or contact me (sihaohu@gatech.edu), and I will reply as soon as I see the issue or email.

