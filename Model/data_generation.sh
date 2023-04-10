
python gen_seq.py --dup=True --deanon=True --phisher=True --tornado=True --dataset=1M --bizdate=bert4eth_1M_min3_dup

python gen_pretrain_data.py --source_bizdate=bert4eth_1M_min3_dup --bizdate=bert4eth_1M_min3_dup_seq100_mask80 --max_seq_length=100 --masked_lm_prob=0.8

# generate finetune data for phishing account detection

python gen_finetune_phisher_data.py --bizdate=bert4eth_1M_min3_dup_seq100_mask80 --source_bizdate=bert4eth_1M_min3_dup --max_seq_length=100
