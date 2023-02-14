


python gen_pretrain_data_ERC.py --source_bizdate=bert4eth_1M_min3_dup_ERC --bizdate=bert4eth_1M_min3_dup_seq100_mask80_ERC --max_seq_length=100 --masked_lm_prob=0.8



CUDA_VISIBLE_DEVICES=7 python run_pretrain.py --bizdate=bert4eth_1M_min3_dup_seq100_mask80_ERC --max_seq_length=100 --checkpointDir=bert4eth_1M_min3_dup_seq100_mask80_ERC_zipfan5000 --neg_sample_num=5000 --neg_strategy=zip


# output embedding

CUDA_VISIBLE_DEVICES=0 python run_embed.py --bizdate=bert4eth_1M_min3_dup_seq100_mask80_ERC --init_checkpoint=bert4eth_1M_min3_dup_seq100_mask80_ERC_zipfan5000/model_72000  --max_seq_length=100 --neg_sample_num=5000 --neg_strategy=zip

CUDA_VISIBLE_DEVICES=1 python run_embed.py --bizdate=bert4eth_1M_min3_dup_seq100_mask80_ERC --init_checkpoint=bert4eth_1M_min3_dup_seq100_mask80_ERC_zipfan5000/model_96000  --max_seq_length=100 --neg_sample_num=5000 --neg_strategy=zip


#

#

CUDA_VISIBLE_DEVICES=3 python run_finetune_phisher.py --bizdate=bert4eth_1M_min3_dup_seq100_mask80 --max_seq_length=100 --checkpointDir=tmp


CUDA_VISIBLE_DEVICES=0 python run_finetune_phisher.py --bizdate=bert4eth_1M_min3_dup_seq100_mask80_ERC --max_seq_length=100 --checkpointDir=tmp --init_checkpoint=bert4eth_1M_min3_dup_seq100_mask80_ERC_zipfan5000/model_72000

CUDA_VISIBLE_DEVICES=3 python run_finetune_phisher.py --bizdate=bert4eth_1M_min3_dup_seq100_mask80_ERC --max_seq_length=100 --checkpointDir=tmp


--init_checkpoint=bert4eth_1M_min3_dup_seq100_mask80_ERC_zipfan5000/model_72000


bert4eth_1M_min3_dup_seq100_mask80_ERC