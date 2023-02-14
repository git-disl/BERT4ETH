

# BERT4ETH experiment


# share 5000 zip
CUDA_VISIBLE_DEVICES=0 python run_pretrain.py --bizdate=bert4eth_1M_min3_dup_seq100_mask80_IOS --max_seq_length=100 --checkpointDir=bert4eth_1M_min3_dup_seq100_mask80_IOS_zipfan5000 --neg_sample_num=5000 --neg_strategy=zip

# output embedding

CUDA_VISIBLE_DEVICES=0 python run_embed.py --bizdate=bert4eth_1M_min3_dup_seq100_mask80_IOS --init_checkpoint=bert4eth_1M_min3_dup_seq100_mask80_IOS_zipfan5000/model_72000  --max_seq_length=100 --neg_sample_num=5000 --neg_strategy=zip

CUDA_VISIBLE_DEVICES=1 python run_embed.py --bizdate=bert4eth_1M_min3_dup_seq100_mask80_IOS --init_checkpoint=bert4eth_1M_min3_dup_seq100_mask80_IOS_zipfan5000/model_96000  --max_seq_length=100 --neg_sample_num=5000 --neg_strategy=zip

# run_finetune

CUDA_VISIBLE_DEVICES=3 python run_finetune_phisher.py --max_seq_length=100 --checkpointDir=tmp --bizdate=bert4eth_1M_min3_dup_seq100_mask80_IOS


--init_checkpoint=bert4eth_1M_min3_dup_seq100_mask80_IOS_zipfan5000/model_72000
