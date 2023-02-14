

# BERT4ETH experiment

# shared实验

# share 1000 zip
CUDA_VISIBLE_DEVICES=0 python run_pretrain.py --bizdate=bert4eth_1M_min3_dup_seq100_mask80 --max_seq_length=100 --checkpointDir=bert4eth_1M_min3_dup_seq100_mask80_shared_zipfan1000 --neg_sample_num=1000 --neg_strategy=zip --neg_share=True

# share 5000 zip
CUDA_VISIBLE_DEVICES=1 python run_pretrain.py --bizdate=bert4eth_1M_min3_dup_seq100_mask80 --max_seq_length=100 --checkpointDir=bert4eth_1M_min3_dup_seq100_mask80_shared_zipfan5000 --neg_sample_num=5000 --neg_strategy=zip --neg_share=True

# share 10000 zip
CUDA_VISIBLE_DEVICES=2 python run_pretrain.py --bizdate=bert4eth_1M_min3_dup_seq100_mask80 --max_seq_length=100 --checkpointDir=bert4eth_1M_min3_dup_seq100_mask80_shared_zipfan10000 --neg_sample_num=10000 --neg_strategy=zip --neg_share=True

# share 1000 uniform

CUDA_VISIBLE_DEVICES=3 python run_pretrain.py --bizdate=bert4eth_1M_min3_dup_seq100_mask80 --max_seq_length=100 --checkpointDir=bert4eth_1M_min3_dup_seq100_mask80_shared_uniform1000 --neg_sample_num=1000 --neg_strategy=uniform --neg_share=True

# share 5000 zip
CUDA_VISIBLE_DEVICES=4 python run_pretrain.py --bizdate=bert4eth_1M_min3_dup_seq100_mask80 --max_seq_length=100 --checkpointDir=bert4eth_1M_min3_dup_seq100_mask80_shared_uniform5000 --neg_sample_num=5000 --neg_strategy=uniform --neg_share=True

# share 10000 zip
CUDA_VISIBLE_DEVICES=5 python run_pretrain.py --bizdate=bert4eth_1M_min3_dup_seq100_mask80 --max_seq_length=100 --checkpointDir=bert4eth_1M_min3_dup_seq100_mask80_shared_uniform10000 --neg_sample_num=10000 --neg_strategy=uniform --neg_share=True


# share 1000 freq (1.0)
CUDA_VISIBLE_DEVICES=6 python run_pretrain.py --bizdate=bert4eth_1M_min3_dup_seq100_mask80 --max_seq_length=100 --checkpointDir=bert4eth_1M_min3_dup_seq100_mask80_shared_freq10_1000 --neg_sample_num=1000 --neg_strategy=freq --neg_share=True


# share 5000 freq (1.0)
CUDA_VISIBLE_DEVICES=7 python run_pretrain.py --bizdate=bert4eth_1M_min3_dup_seq100_mask80 --max_seq_length=100 --checkpointDir=bert4eth_1M_min3_dup_seq100_mask80_shared_freq10_5000 --neg_sample_num=5000 --neg_strategy=freq --neg_share=True


# share 10000 freq (1.0)
CUDA_VISIBLE_DEVICES=0 python run_pretrain.py --bizdate=bert4eth_1M_min3_dup_seq100_mask80 --max_seq_length=100 --checkpointDir=bert4eth_1M_min3_dup_seq100_mask80_shared_freq10_10000 --neg_sample_num=10000 --neg_strategy=freq --neg_share=True


# share 1000 freq (0.5)
CUDA_VISIBLE_DEVICES=1 python run_pretrain.py --bizdate=bert4eth_1M_min3_dup_seq100_mask80 --max_seq_length=100 --checkpointDir=bert4eth_1M_min3_dup_seq100_mask80_shared_freq05_1000 --neg_sample_num=1000 --neg_strategy=freq --neg_share=True


# share 5000 freq (0.5)
CUDA_VISIBLE_DEVICES=2 python run_pretrain.py --bizdate=bert4eth_1M_min3_dup_seq100_mask80 --max_seq_length=100 --checkpointDir=bert4eth_1M_min3_dup_seq100_mask80_shared_freq05_5000 --neg_sample_num=5000 --neg_strategy=freq --neg_share=True



# share 10000 freq (0.5)
CUDA_VISIBLE_DEVICES=3 python run_pretrain.py --bizdate=bert4eth_1M_min3_dup_seq100_mask80 --max_seq_length=100 --checkpointDir=bert4eth_1M_min3_dup_seq100_mask80_shared_freq05_10000 --neg_sample_num=10000 --neg_strategy=freq --neg_share=True

# 先看一下unshare 80的结果 uniform, zipfan
# unshare


# unshare 20 zip
CUDA_VISIBLE_DEVICES=4 python run_pretrain.py --bizdate=bert4eth_1M_min3_dup_seq100_mask80 --max_seq_length=100 --checkpointDir=bert4eth_1M_min3_dup_seq100_mask80_unshared_zipfan20 --neg_sample_num=20 --neg_strategy=zip --neg_share=False

# unshare 20 uniform
CUDA_VISIBLE_DEVICES=5 python run_pretrain.py --bizdate=bert4eth_1M_min3_dup_seq100_mask80 --max_seq_length=100 --checkpointDir=bert4eth_1M_min3_dup_seq100_mask80_unshared_uniform20 --neg_sample_num=20 --neg_strategy=uniform --neg_share=False

# unshare 20 freq 1.0
CUDA_VISIBLE_DEVICES=6 python run_pretrain.py --bizdate=bert4eth_1M_min3_dup_seq100_mask80 --max_seq_length=100 --checkpointDir=bert4eth_1M_min3_dup_seq100_mask80_unshared_freq10_20 --neg_sample_num=20 --neg_strategy=freq --neg_share=False

# unshare 20 freq 0.5
CUDA_VISIBLE_DEVICES=7 python run_pretrain.py --bizdate=bert4eth_1M_min3_dup_seq100_mask80 --max_seq_length=100 --checkpointDir=bert4eth_1M_min3_dup_seq100_mask80_unshared_freq05_20 --neg_sample_num=20 --neg_strategy=freq --neg_share=False


# masking 实验。
# unshare 20 uniform
CUDA_VISIBLE_DEVICES=0 python run_pretrain.py --bizdate=bert4eth_1M_min3_dup_seq100_mask10 --max_seq_length=100 --checkpointDir=bert4eth_1M_min3_dup_seq100_mask10_unshared_uniform20 --masked_lm_prob=0.1 --neg_sample_num=20 --neg_strategy=uniform --neg_share=False

CUDA_VISIBLE_DEVICES=1 python run_pretrain.py --bizdate=bert4eth_1M_min3_dup_seq100_mask15 --max_seq_length=100 --checkpointDir=bert4eth_1M_min3_dup_seq100_mask15_unshared_uniform20 --masked_lm_prob=0.15 --neg_sample_num=20 --neg_strategy=uniform --neg_share=False

CUDA_VISIBLE_DEVICES=2 python run_pretrain.py --bizdate=bert4eth_1M_min3_dup_seq100_mask20 --max_seq_length=100 --checkpointDir=bert4eth_1M_min3_dup_seq100_mask20_unshared_uniform20 --masked_lm_prob=0.2 --neg_sample_num=20 --neg_strategy=uniform --neg_share=False

CUDA_VISIBLE_DEVICES=3 python run_pretrain.py --bizdate=bert4eth_1M_min3_dup_seq100_mask30 --max_seq_length=100 --checkpointDir=bert4eth_1M_min3_dup_seq100_mask30_unshared_uniform20 --masked_lm_prob=0.3 --neg_sample_num=20 --neg_strategy=uniform --neg_share=False


CUDA_VISIBLE_DEVICES=4 python run_pretrain.py --bizdate=bert4eth_1M_min3_dup_seq100_mask40 --max_seq_length=100 --checkpointDir=bert4eth_1M_min3_dup_seq100_mask40_unshared_uniform20 --masked_lm_prob=0.4 --neg_sample_num=20 --neg_strategy=uniform --neg_share=False

CUDA_VISIBLE_DEVICES=5 python run_pretrain.py --bizdate=bert4eth_1M_min3_dup_seq100_mask50 --max_seq_length=100 --checkpointDir=bert4eth_1M_min3_dup_seq100_mask50_unshared_uniform20 --masked_lm_prob=0.5 --neg_sample_num=20 --neg_strategy=uniform --neg_share=False

CUDA_VISIBLE_DEVICES=6 python run_pretrain.py --bizdate=bert4eth_1M_min3_dup_seq100_mask60 --max_seq_length=100 --checkpointDir=bert4eth_1M_min3_dup_seq100_mask60_unshared_uniform20 --masked_lm_prob=0.6 --neg_sample_num=20 --neg_strategy=uniform --neg_share=False

CUDA_VISIBLE_DEVICES=7 python run_pretrain.py --bizdate=bert4eth_1M_min3_dup_seq100_mask70 --max_seq_length=100 --checkpointDir=bert4eth_1M_min3_dup_seq100_mask70_unshared_uniform20 --masked_lm_prob=0.7 --neg_sample_num=20 --neg_strategy=uniform --neg_share=False

CUDA_VISIBLE_DEVICES=0 python run_pretrain.py --bizdate=bert4eth_1M_min3_dup_seq100_mask90 --max_seq_length=100 --checkpointDir=bert4eth_1M_min3_dup_seq100_mask90_unshared_uniform20 --masked_lm_prob=0.9 --neg_sample_num=20 --neg_strategy=uniform --neg_share=False

# output embedding 进行测试,

# 先确定index

# zipfan 5000 share

CUDA_VISIBLE_DEVICES=1 python run_embed.py --bizdate=bert4eth_1M_min3_dup_seq100_mask80 --init_checkpoint=bert4eth_1M_min3_dup_seq100_mask80_shared_zipfan5000/model_104000  --max_seq_length=100 --neg_sample_num=5000 --neg_strategy=zip --neg_share=True

CUDA_VISIBLE_DEVICES=1 python run_embed.py --bizdate=bert4eth_1M_min3_dup_seq100_mask80 --init_checkpoint=bert4eth_1M_min3_dup_seq100_mask80_shared_zipfan5000/model_96000  --max_seq_length=100 --neg_sample_num=5000 --neg_strategy=zip --neg_share=True
CUDA_VISIBLE_DEVICES=1 python run_embed.py --bizdate=bert4eth_1M_min3_dup_seq100_mask80 --init_checkpoint=bert4eth_1M_min3_dup_seq100_mask80_shared_zipfan5000/model_88000  --max_seq_length=100 --neg_sample_num=5000 --neg_strategy=zip --neg_share=True
CUDA_VISIBLE_DEVICES=1 python run_embed.py --bizdate=bert4eth_1M_min3_dup_seq100_mask80 --init_checkpoint=bert4eth_1M_min3_dup_seq100_mask80_shared_zipfan5000/model_72000  --max_seq_length=100 --neg_sample_num=5000 --neg_strategy=zip --neg_share=True
CUDA_VISIBLE_DEVICES=1 python run_embed.py --bizdate=bert4eth_1M_min3_dup_seq100_mask80 --init_checkpoint=bert4eth_1M_min3_dup_seq100_mask80_shared_zipfan5000/model_64000  --max_seq_length=100 --neg_sample_num=5000 --neg_strategy=zip --neg_share=True


for index in {"64000","72000","88000"};
do
CUDA_VISIBLE_DEVICES=0 python run_embed.py --bizdate=bert4eth_1M_min3_dup_seq100_mask80 --init_checkpoint=bert4eth_1M_min3_dup_seq100_mask80_shared_freq05_10000/model_$index  --max_seq_length=100 --neg_sample_num=10000 --neg_strategy=freq --neg_share=True

done;

for index in {"64000","72000","88000"};
do
CUDA_VISIBLE_DEVICES=1 python run_embed.py --bizdate=bert4eth_1M_min3_dup_seq100_mask80 --init_checkpoint=bert4eth_1M_min3_dup_seq100_mask80_shared_freq05_5000/model_$index  --max_seq_length=100 --neg_sample_num=5000 --neg_strategy=freq --neg_share=True
done;

for index in {"64000","72000","88000"};
do
CUDA_VISIBLE_DEVICES=2 python run_embed.py --bizdate=bert4eth_1M_min3_dup_seq100_mask80 --init_checkpoint=bert4eth_1M_min3_dup_seq100_mask80_shared_freq05_1000/model_$index  --max_seq_length=100 --neg_sample_num=1000 --neg_strategy=freq --neg_share=True
done;


for index in {"64000","72000","88000"};
do
CUDA_VISIBLE_DEVICES=3 python run_embed.py --bizdate=bert4eth_1M_min3_dup_seq100_mask80 --init_checkpoint=bert4eth_1M_min3_dup_seq100_mask80_unshared_freq05_20/model_$index  --max_seq_length=100 --neg_sample_num=20 --neg_strategy=freq --neg_share=False
done;



CUDA_VISIBLE_DEVICES=0 python run_embed.py --bizdate=bert4eth_1M_min3_dup_seq100_mask10 --masked_lm_prob=0.1 --init_checkpoint=bert4eth_1M_min3_dup_seq100_mask10_unshared_uniform20/model_72000  --max_seq_length=100 --neg_sample_num=20 --neg_strategy=uniform --neg_share=False

CUDA_VISIBLE_DEVICES=0 python run_embed.py --bizdate=bert4eth_1M_min3_dup_seq100_mask15 --masked_lm_prob=0.15 --init_checkpoint=bert4eth_1M_min3_dup_seq100_mask15_unshared_uniform20/model_72000  --max_seq_length=100 --neg_sample_num=20 --neg_strategy=uniform --neg_share=False

CUDA_VISIBLE_DEVICES=0 python run_embed.py --bizdate=bert4eth_1M_min3_dup_seq100_mask15 --masked_lm_prob=0.15 --init_checkpoint=bert4eth_1M_min3_dup_seq100_mask15_unshared_uniform20/model_96000  --max_seq_length=100 --neg_sample_num=20 --neg_strategy=uniform --neg_share=False


CUDA_VISIBLE_DEVICES=0 python run_embed.py --bizdate=bert4eth_1M_min3_dup_seq100_mask20 --masked_lm_prob=0.2 --init_checkpoint=bert4eth_1M_min3_dup_seq100_mask20_unshared_uniform20/model_72000  --max_seq_length=100 --neg_sample_num=20 --neg_strategy=uniform --neg_share=False


CUDA_VISIBLE_DEVICES=1 python run_embed.py --bizdate=bert4eth_1M_min3_dup_seq100_mask30 --masked_lm_prob=0.3 --init_checkpoint=bert4eth_1M_min3_dup_seq100_mask30_unshared_uniform20/model_72000  --max_seq_length=100 --neg_sample_num=20 --neg_strategy=uniform --neg_share=False
CUDA_VISIBLE_DEVICES=1 python run_embed.py --bizdate=bert4eth_1M_min3_dup_seq100_mask40 --masked_lm_prob=0.4 --init_checkpoint=bert4eth_1M_min3_dup_seq100_mask40_unshared_uniform20/model_72000  --max_seq_length=100 --neg_sample_num=20 --neg_strategy=uniform --neg_share=False
CUDA_VISIBLE_DEVICES=1 python run_embed.py --bizdate=bert4eth_1M_min3_dup_seq100_mask50 --masked_lm_prob=0.5 --init_checkpoint=bert4eth_1M_min3_dup_seq100_mask50_unshared_uniform20/model_72000  --max_seq_length=100 --neg_sample_num=20 --neg_strategy=uniform --neg_share=False


CUDA_VISIBLE_DEVICES=2 python run_embed.py --bizdate=bert4eth_1M_min3_dup_seq100_mask60 --masked_lm_prob=0.6 --init_checkpoint=bert4eth_1M_min3_dup_seq100_mask60_unshared_uniform20/model_72000  --max_seq_length=100 --neg_sample_num=20 --neg_strategy=uniform --neg_share=False
CUDA_VISIBLE_DEVICES=2 python run_embed.py --bizdate=bert4eth_1M_min3_dup_seq100_mask70 --masked_lm_prob=0.7 --init_checkpoint=bert4eth_1M_min3_dup_seq100_mask70_unshared_uniform20/model_72000  --max_seq_length=100 --neg_sample_num=20 --neg_strategy=uniform --neg_share=False
CUDA_VISIBLE_DEVICES=2 python run_embed.py --bizdate=bert4eth_1M_min3_dup_seq100_mask90 --masked_lm_prob=0.9 --init_checkpoint=bert4eth_1M_min3_dup_seq100_mask90_unshared_uniform20/model_72000  --max_seq_length=100 --neg_sample_num=20 --neg_strategy=uniform --neg_share=False


# finetune实验
CUDA_VISIBLE_DEVICES=3 python run_finetune_phisher.py --bizdate=bert4eth_1M_min3_dup_seq100_mask80 --max_seq_length=100 --checkpointDir=tmp

--init_checkpoint=bert4eth_1M_min3_dup_seq100_mask80_shared_zipfan5000/model_64000

# 原始BERT finetune实验
# 先要生成finetune数据

CUDA_VISIBLE_DEVICES=5 python run_finetune_phisher.py --bizdate=bert4eth_1M_min3_dup_seq100_mask15 --max_seq_length=100 --checkpointDir=tmp --init_checkpoint=bert4eth_1M_min3_dup_seq100_mask15_unshared_uniform20/model_72000 --masked_lm_prob=0.15 --neg_sample_num=20 --neg_share=False --neg_strategy=uniform



CUDA_VISIBLE_DEVICES=2 python run_finetune_phisher.py --bizdate=bert4eth_1M_min3_dup_seq100_mask80 --max_seq_length=100 --checkpointDir=tmp --init_checkpoint=bert4eth_1M_min3_dup_seq100_mask80_shared_zipfan5000/model_72000

CUDA_VISIBLE_DEVICES=2 python run_finetune_phisher.py --bizdate=bert4eth_1M_min3_dup_seq100_mask80 --max_seq_length=100 --checkpointDir=tmp --init_checkpoint=bert4eth_1M_min3_dup_seq100_mask80_shared_zipfan5000/model_88000

CUDA_VISIBLE_DEVICES=3 python run_finetune_phisher.py --bizdate=bert4eth_1M_min3_dup_seq100_mask80 --max_seq_length=100 --checkpointDir=tmp --init_checkpoint=bert4eth_1M_min3_dup_seq100_mask80_shared_zipfan5000/model_104000

CUDA_VISIBLE_DEVICES=2 python run_finetune_phisher.py --bizdate=bert4eth_1M_min3_dup_seq100_mask80 --max_seq_length=100 --checkpointDir=tmp


--init_checkpoint=bert4eth_1M_min3_dup_seq100_mask80_shared_zipfan5000/model_120000

# ablation experiment (w\o features)

CUDA_VISIBLE_DEVICES=0 python run_pretrain.py --bizdate=bert4eth_1M_min3_dup_seq100_mask80 --max_seq_length=100 --checkpointDir=bert4eth_1M_min3_dup_seq100_mask80_shared_zipfan5000_nofeature --neg_sample_num=5000 --neg_strategy=zip --neg_share=True
# output embedding

CUDA_VISIBLE_DEVICES=1 python run_embed.py --bizdate=bert4eth_1M_min3_dup_seq100_mask80 --init_checkpoint=bert4eth_1M_min3_dup_seq100_mask80_shared_zipfan5000_nofeature/model_72000  --max_seq_length=100 --neg_sample_num=5000 --neg_strategy=zip --neg_share=True



# ablation on no duplicated sequence
CUDA_VISIBLE_DEVICES=1 python run_pretrain.py --bizdate=bert4eth_1M_min3_nodup_seq100_mask80 --max_seq_length=100 --checkpointDir=bert4eth_1M_min3_nodup_seq100_mask80_shared_zipfan5000 --neg_sample_num=5000 --neg_strategy=zip --neg_share=True
# output embedding
CUDA_VISIBLE_DEVICES=1 python run_embed.py --bizdate=bert4eth_1M_min3_nodup_seq100_mask80 --init_checkpoint=bert4eth_1M_min3_nodup_seq100_mask80_shared_zipfan5000/model_96000  --max_seq_length=100 --neg_sample_num=5000 --neg_strategy=zip --neg_share=True


# ablation on 80% masking ratio

CUDA_VISIBLE_DEVICES=6 python run_pretrain.py --bizdate=bert4eth_1M_min3_dup_seq100_mask15 --max_seq_length=100 --checkpointDir=bert4eth_1M_min3_dup_seq100_mask15_shared_zipfan5000 --masked_lm_prob=0.15 --neg_sample_num=5000 --neg_strategy=zip --neg_share=True

CUDA_VISIBLE_DEVICES=0 python run_embed.py --bizdate=bert4eth_1M_min3_dup_seq100_mask15 --masked_lm_prob=0.15 --init_checkpoint=bert4eth_1M_min3_dup_seq100_mask15_shared_zipfan5000/model_72000  --max_seq_length=100 --neg_sample_num=5000 --neg_strategy=zip --neg_share=True
