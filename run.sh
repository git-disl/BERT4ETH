
python gen_seq.py --bizdate=bert4eth_exp

python gen_pretrain_data.py --bizdate=bert4eth_exp  \
                            --max_seq_length=100  \
                            --dupe_factor=10 \
                            --masked_lm_prob=0.8

CUDA_VISIBLE_DEVICES=3 python run_pretrain.py --bizdate=bert4eth_exp \
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

# output representations
CUDA_VISIBLE_DEVICES=3 python output_embed.py --bizdate=bert4eth_exp \
                    --init_checkpoint=bert4eth_exp/model_104000 \
                    --max_seq_length=100 \
                    --neg_sample_num=5000 \
                    --neg_strategy=zip \
                    --neg_share=True


# Testing on the account representation
python run_phishing_detection.py --init_checkpoint=bert4eth_exp/model_104000

CUDA_VISIBLE_DEVICES=3 python run_phishing_detection_dnn.py --init_checkpoint=bert4eth_exp/model_104000

# generate finetune data for phishing account detection
python gen_finetune_phisher_data.py --bizdate=bert4eth_exp \
                                    --max_seq_length=100

CUDA_VISIBLE_DEVICES=3 python run_finetune_phisher.py --init_checkpoint=bert4eth_exp/model_104000 \
                                                      --bizdate=bert4eth_exp \
                                                      --max_seq_length=100 \
                                                      --checkpointDir=tmp

