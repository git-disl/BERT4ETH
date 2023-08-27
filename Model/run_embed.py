# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run masked LM/next sentence masked_lm pre-training for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
sys.path.append("")
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import time
from run_pretrain import *

def del_flags(FLAGS, keys_list):
    for keys in keys_list:
        FLAGS.__delattr__(keys)
    return

def main(_):

    mode = tf.estimator.ModeKeys.EVAL
    input_files = FLAGS.test_input_file + "." + FLAGS.bizdate
    features = input_fn(input_files, is_training=False)

    # modeling
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    tf.gfile.MakeDirs(FLAGS.checkpointDir)

    # load vocab
    vocab_file_name = FLAGS.data_dir + FLAGS.vocab_filename + "." + FLAGS.bizdate
    with open(vocab_file_name, "rb") as f:
        vocab = pkl.load(f)

    # must have checkpoint
    if FLAGS.init_checkpoint==None:
        raise ValueError("Must need a checkpoint for evaluation")

    bert_model, total_loss = model_fn(features, mode, bert_config, vocab, FLAGS.init_checkpoint, FLAGS.learning_rate,
                                      FLAGS.num_train_steps, FLAGS.num_warmup_steps, False, False)

    sequence_output = bert_model.get_sequence_output()
    address_output = tf.nn.embedding_lookup(bert_model.get_embedding_table(), features["address"])

    # start session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        address_id_list = []
        sequence_output_vector_list = []
        address_output_vector_list = []

        iter = 0
        start = time.time()
        while True:

            try:
                address_id_v, sequence_output_v, address_output_v = sess.run([features["address"], sequence_output, address_output])
                sequence_output_vector_list.append(sequence_output_v[:, 0, :])
                address_output_vector_list.append(np.squeeze(address_output_v))
                address_id_list.append(np.squeeze(address_id_v))

                if iter % 500 == 0:
                    end = time.time()
                    print("iter=%d, time=%.2fs" % (iter, end - start))
                    start = time.time()

                iter += 1

            except Exception as e:
                print("Out of Sequence")
                # print(e)
                # save model
                # saver.save(sess, os.path.join(FLAGS.checkpointDir, "model_" + str(iter)))
                break

        print("========Embedding Generation Results==========")
        sequence_output_vector_list = np.concatenate(sequence_output_vector_list, axis=0)
        # address_output_vector_list = np.concatenate(address_output_vector_list, axis=0)

        address_id_list = np.concatenate(address_id_list, axis=0)
        address_list = vocab.convert_ids_to_tokens(address_id_list)

        print("sample_num=%d" % (sequence_output_vector_list.shape[0]))
        # write to file

        print("saving embedding and address..")
        checkpoint_name = FLAGS.init_checkpoint.split("/")[0]
        model_index = str(FLAGS.init_checkpoint.split("/")[-1].split("_")[1])
        embed_output_file = "./data/embedding_" + checkpoint_name + "_" + model_index + ".npy"
        print(embed_output_file)
        np.save(embed_output_file, sequence_output_vector_list)
        address_output_file = "./data/address_" + checkpoint_name + "_" + model_index + ".npy"
        print(address_output_file)
        np.save(address_output_file, address_list)

    return

if __name__ == '__main__':

    del_flags(FLAGS, ["do_train", "do_eval", "test_input_file", "neg_sample_num", "init_checkpoint"])
    flags.DEFINE_bool("do_train", False, "")
    flags.DEFINE_bool("do_eval", True, "")
    flags.DEFINE_bool("attention_output", False, "")
    flags.DEFINE_string("test_input_file", "./inter_data/embed.tfrecord", "Example for embedding generation.")
    flags.DEFINE_integer("neg_sample_num", 5000, "The number of negative samples in a batch")
    flags.DEFINE_string("init_checkpoint", None, "Initial checkpoint (usually from a pre-trained BERT model).")

    tf.app.run()