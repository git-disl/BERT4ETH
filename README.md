# BERT4ETH

This is the code and data of the paper "BERT4ETH: A Pre-trained Transformer for Ethereum Fraud Detection", accepted by the Web conference (WWW) 2023.


## Getting Start

### Requirements
* Python >= 3.6.1
* NumPy >= 1.12.1
* TensorFlow >= 1.4.0

### Preprocess dataset 

#### Step 1: Download dataset from Google Drive. 
* Transaction Dataset:
* * [Phishing Account](https://drive.google.com/file/d/11UAhLOcffzLyPhdsIqRuFsJNSqNvrNJf/view?usp=sharing)

* * [De-anonymization](https://drive.google.com/file/d/1Yveis90jCx-nIA6pUL_4SUezMsVJr8dp/view?usp=sharing)

* * [MEV Bot](https://drive.google.com/file/d/10br9Xki_E443MJzGzQHQqLGds-uuGTRU/view?usp=sharing)

* * [Normal Account](https://drive.google.com/file/d/1-htLUymg1UxDrXcI8tslU9wbn0E1vl9_/view?usp=sharing)

* [ERC-20 Log Dataset (all in one)](https://drive.google.com/file/d/1mB2Tf7tMq5ApKKOVdctaTh2UZzzrAVxq/view?usp=sharing)

#### Step 2: Unzip dataset under the directory of "BERT4ETH/Data/" 


```sh
cd BERT4ETH/Data;
unzip ...;
``` 
The total volume of unzipped dataset is quite huge (more than 15GB).

#### Step 3: Transaction Sequence Generation

(In Step 3 we apply the transaction de-duplication strategy.)

```sh
cd Model/bert4eth;
python gen_seq.py --phisher=True \
                  --deanon=True \ 
                  --mev=True \ 
                  --bizdate=xxx
                  
python gen_seq_erc20.py;
``` 

### Pre-training

#### Step 0: Model Configuration

The configuration file is "Model/bert4eth/bert_config_eth_64.json"
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
python gen_pretrain_data.py --bizdate=xxx \
                            --max_seq_length=50 \
                            --masked_lm_prob=0.8 \
                            --max_predictions_per_seq=40 \
                            --sliding_step=30 \
                            --dupe_factor=10 \
                            --do_eval=False
```

#### Step 2: Pre-train BERT4ETH Model

```sh
python run_pretrain.py --bizdate=xxx \
                       --max_seq_length=50 \
                       --max_predictions_per_seq=40 \
                       --masked_lm_prob=0.8 \
                       --epoch=5 \
                       --batch_size=256 \
                       --learning_rate=1e-4 \
                       --num_train_steps=1000000 \
                       --num_warmup_steps=100 \
                       --save_checkpoints_steps=8000 \
                       --neg_strategy=zip \
                       --neg_sample_num=5000 \
                       --do_eval=False \
                       --checkpointDir=xxx \
                       --init_seed=1234 
```

| Parameter                  | Description                                                                        |
|----------------------------|------------------------------------------------------------------------------------|
| `bizdate`                  | The signature for this experiment run.                                             |
| `max_seq_length`           | The maximum length of BERT4ETH.                                                    |
| `max_predictions_per_seq`  | The maximum number of masked addresses in one sequence.                            |
| `masked_lm_prob`           | The probability of masking an address.                                             |
| `epochs`                   | Number of training epochs, default = `5`.                                          |
| `batch_size`               | Batch size, default = `256`.                                                       |
| `learning_rate`            | Learning rate for the optimizer (Adam), default = `1e-4`.                          |
| `num_train_steps`          | The maximum number of training steps, default = `1000000`,                         |
| `num_warmup_steps`         | The step number for warm-up training, default = `100`.                             |
| `save_checkpoints_steps`   | The parameter controlling the step of saving checkpoints, default = `8000`.        |
| `neg_strategy`             | Strategy for negative sampling, default `zip`, options (`uniform`, `zip`, `freq`). |
| `neg_sample_num`           | The negative sampling number for one batch, default = `5000`.                      |
| `do_eval`                  | Whether to do evaluation during training, default = `False`.                       |
| `checkpointDir`            | Specify the directory to save the checkpoints.                                     |
| `init_seed`                | The initial seed, default = `1234`.                                                |



#### Step 3: Output Representation

```sh
python run_embed.py --bizdate=xxx \ 
                    --init_checkpoint=xxx/xxx \ 
                    --neg_strategy=zip \
                    --neg_sample_num=5000 \
                    --do_eval=True \
```

### Testing:

#### Phshing Account Detection
```sh
cd BERT4ETH/Model;
python run_dean.py --metric=euclidean \
                   --algo=bert4eth
``` 

#### De-anonymization

```sh
cd BERT4ETH/Model;
python run_phisher.py --algo=bert4eth
``` 

#### MEV Bot Detection

```sh
cd BERT4ETH/Model;
python run_mev_bot.py --algo=bert4eth
``` 
