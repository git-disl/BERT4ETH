# BERT4ETH

This is the code and data of the paper [BERT4ETH: A Pre-trained Transformer for Ethereum Fraud Detection](https://arxiv.org/abs/2303.18138), accepted by the ACM Web conference (WWW) 2023.


## Getting Start

### Requirements
* Python >= 3.6.1
* NumPy >= 1.12.1
* TensorFlow >= 1.4.0

### Preprocess dataset 

#### Step 1: Download dataset from Google Drive. 
* Transaction Dataset:
* * [Phishing Account](https://drive.google.com/file/d/11UAhLOcffzLyPhdsIqRuFsJNSqNvrNJf/view?usp=sharing)

* * [De-anonymization(ENS)](https://drive.google.com/file/d/1Yveis90jCx-nIA6pUL_4SUezMsVJr8dp/view?usp=sharing)

* * [De-anonymization(Tornado)](https://drive.google.com/file/d/1DMbPSZMSvTYMKUZg3oYKFrjPo2_jeeG4/view?usp=sharing)

* * [Normal Account](https://drive.google.com/file/d/1-htLUymg1UxDrXcI8tslU9wbn0E1vl9_/view?usp=sharing)

* [ERC-20 Log Dataset (all in one)](https://drive.google.com/file/d/1mB2Tf7tMq5ApKKOVdctaTh2UZzzrAVxq/view?usp=sharing)

#### Step 2: Unzip dataset under the directory of "BERT4ETH/Data/" 


```sh
cd BERT4ETH/Data; # Labels are already included
unzip ...;
``` 
The total volume of unzipped dataset is quite huge (more than 15GB).

If you want to run the basic BERT4ETH model, it is no need to download the ERC-20 log dataset.

Advanced features (In/out separation and ERC20 log) make the model not very efficiency..


#### Step 3: Transaction Sequence Generation

(In Step 3 we apply the transaction de-duplication strategy.)

```sh
cd Model/bert4eth;
python gen_seq.py --phisher=True \
                  --deanon=True \ 
                  --mev=True \
                  --dup=True \
                  --dataset=1M \
                  --bizdate=bert4eth_1M_min3_dup

```


### Pre-training

#### Step 0: Model Configuration

The configuration file is "Model/BERT4ETH/bert_config.json"
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
  "num_hidden_layers": 2,
  "type_vocab_size": 2,
  "vocab_size": 3000000
}
```

#### Step 1: Pre-train Sequence Generation 

[//]: # (&#40;Masking, I/O separation and ERC20 log&#41;)

```sh
python gen_pretrain_data.py --source_bizdate=bert4eth_1M_min3_dup  \
                            --bizdate=bert4eth_1M_min3_dup_seq100_mask80  \ 
                            --max_seq_length=100  \
                            --dupe_factor=10 \
                            --masked_lm_prob=0.8 \
                            --do_eval=False
```


#### Step 2: Pre-train BERT4ETH Model

```sh
python run_pretrain.py --bizdate=bert4eth_1M_min3_dup_seq100_mask80 \
                       --max_seq_length=100 \
                       --checkpointDir=bert4eth_1M_min3_dup_seq100_mask80_shared_zipfan5000 \
                       --epoch=5 \
                       --batch_size=256 \
                       --learning_rate=1e-4 \
                       --num_train_steps=1000000 \
                       --num_warmup_steps=100 \
                       --save_checkpoints_steps=8000 \
                       --neg_strategy=zip
                       --neg_sample_num=5000 
                       --neg_share=True
                       --init_seed=1234 
                       
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
| `num_warmup_steps`       | The step number for warm-up training, default = `100`.                             |
| `save_checkpoints_steps` | The parameter controlling the step of saving checkpoints, default = `8000`.        |
| `neg_strategy`           | Strategy for negative sampling, default `zip`, options (`uniform`, `zip`, `freq`). |
| `neg_share`              | Whether enable in-batch sharing strategy, default = `True`.                        |
| `neg_sample_num`         | The negative sampling number for one batch, default = `5000`.                      |
| `do_eval`                | Whether to do evaluation during training, default = `False`.                       |
| `checkpointDir`          | Specify the directory to save the checkpoints.                                     |
| `init_seed`              | The initial seed, default = `1234`.                                                |



#### Step 3: Output Representation


```sh
python run_embed.py --bizdate=bert4eth_1M_min3_dup_seq100_mask80 \ 
                    --init_checkpoint=bert4eth_1M_min3_dup_seq100_mask80_shared_zipfan5000/model_104000 \  
                    --max_seq_length=100 \
                    --neg_sample_num=5000 \
                    --neg_strategy=zip \
                    --neg_share=True 
```

### Step 4: Testing on the account representation

#### Phishing Account Detection
```sh
cd BERT4ETH/Model;
python run_phishing_detection.py --algo=bert4eth \
                                 --model_index=XXX

```

#### De-anonymization (ENS dataset)

```sh
cd BERT4ETH/Model;
python run_dean_ENS.py --metric=euclidean \
                       --algo=bert4eth \
                       --model_index=XXX
```

#### De-anonymization (Tornado Cash)

```sh
cd BERT4ETH/Model;
python run_dean_Tornado.py --metric=euclidean \
                           --algo=bert4eth \
                           --model_index=XXX
```

### Fine-tuning on the phishing account detection

```sh
python gen_finetune_phisher_data.py --bizdate=bert4eth_1M_min3_dup_seq100_mask80 \ 
                                    --source_bizdate=bert4eth_1M_min3_dup \
                                    --max_seq_length=100 
```

```sh
python run_finetune_phisher.py --bizdate=bert4eth_1M_min3_dup_seq100_mask80 \ 
                               --max_seq_length=100 --checkpointDir=tmp
```

-----
## Citation

If you find this repository useful, please give us a star and consider citing our paper : ) Thank you!
```
@article{hu2023bert4eth,
  title={BERT4ETH: A Pre-trained Transformer for Ethereum Fraud Detection},
  author={Hu, Sihao and Zhang, Zhen and Luo, Bingqiao and Lu, Shengliang and He, Bingsheng and Liu, Ling},
  journal={arXiv preprint arXiv:2303.18138},
  year={2023}
}
```


