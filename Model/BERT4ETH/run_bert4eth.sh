
# pre-train
CUDA_VISIBLE_DEVICES=2 python run_pretrain.py --bizdate=bert4eth_1M_min3_dup_seq100_mask80 --max_seq_length=100 --checkpointDir=bert4eth_1M_min3_dup_seq100_mask80_shared_zipfan5000 --neg_sample_num=5000 --neg_strategy=zip --neg_share=True

# run_embeding
CUDA_VISIBLE_DEVICES=2 python run_embed.py --bizdate=bert4eth_1M_min3_dup_seq100_mask80 --init_checkpoint=bert4eth_1M_min3_dup_seq100_mask80_shared_zipfan5000/model_104000  --max_seq_length=100 --neg_sample_num=5000 --neg_strategy=zip --neg_share=True

# finetune for phishing account detection
CUDA_VISIBLE_DEVICES=3 python run_finetune_phisher.py --bizdate=bert4eth_1M_min3_dup_seq100_mask80 --max_seq_length=100 --checkpointDir=tmp
