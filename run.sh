
python gen_seq.py --bizdate=bert4eth_exp

python gen_pretrain_data.py --bizdate=bert4eth_exp  \
                            --max_seq_length=100  \
                            --dupe_factor=10 \
                            --masked_lm_prob=0.8

cd BERT4ETH

python run_pretrain.py --bizdate=bert4eth_exp \
                       --max_seq_length=100 \
                       --checkpointDir=bert4eth_exp \
                       --epoch=5 \
                       --batch_size=256 \
                       --learning_rate=1e-4 \
                       --num_train_steps=1000000 \
                       --save_checkpoints_steps=8000 \
                       --neg_strategy=zip \
                       --neg_sample_num=5000 \
                       --neg_share=True


python run_embed.py --bizdate=bert4eth_exp \
                    --init_checkpoint=bert4eth_exp/model_104000 \
                    --max_seq_length=100 \
                    --neg_sample_num=5000 \
                    --neg_strategy=zip \
                    --neg_share=True


# generate finetune data for phishing account detection
python gen_finetune_phisher_data.py --bizdate=bert4eth_tiny \
                                    --max_seq_length=100

