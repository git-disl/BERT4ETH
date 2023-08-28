
# Step 1: generate transaction sequence
python gen_seq.py --bizdate=bert4eth_exp

# Step 2: generate pre-train data (do masking)
python gen_pretrain_data.py --bizdate=bert4eth_exp  \
                            --max_seq_length=100  \
                            --dupe_factor=10 \
                            --masked_lm_prob=0.8

# Step 3: pre-train the BERT4ETH model
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

# Step 4: output representations
CUDA_VISIBLE_DEVICES=3 python output_embed.py --bizdate=bert4eth_exp \
                    --init_checkpoint=bert4eth_exp/model_104000 \
                    --max_seq_length=100 \
                    --neg_sample_num=5000 \
                    --neg_strategy=zip \
                    --neg_share=True


# Phishing detection on the account representation with random forest
python run_phishing_detection.py --init_checkpoint=bert4eth_exp/model_104000

# Phishing detection on the account representation with DNN (better than random forest)
CUDA_VISIBLE_DEVICES=3 python run_phishing_detection_dnn.py --init_checkpoint=bert4eth_exp/model_104000

# Generate finetune data for phishing account detection
python gen_finetune_phisher_data.py --bizdate=bert4eth_exp \
                                    --max_seq_length=100
# Fine-tune the pre-trained model for phishing account detection
CUDA_VISIBLE_DEVICES=3 python run_finetune_phisher.py --init_checkpoint=bert4eth_exp/model_104000 \
                                                      --bizdate=bert4eth_exp \
                                                      --max_seq_length=100 \
                                                      --checkpointDir=tmp

# De-anonymization
python run_dean_ENS.py --metric=euclidean \
                       --init_checkpoint=bert4eth_exp/model_104000