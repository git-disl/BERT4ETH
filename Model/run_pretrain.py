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
import os
import modeling
import sys
sys.path.append("")
import optimization
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np

import math
import pickle as pkl
# import time
from timeit import default_timer as timer

flags = tf.flags
FLAGS = flags.FLAGS

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

## Required parameters
flags.DEFINE_string(
    "bert_config_file", "./bert_config.json",
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "train_input_file", "../inter_data/train.tfrecord",
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "test_input_file", "../inter_data/test.tfrecord",
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "checkpointDir", "ckpt_dir",
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string("signature", 'default', "signature_name")

## Other parameters
flags.DEFINE_string("init_checkpoint", None, "Initial checkpoint (usually from a pre-trained BERT model).")
flags.DEFINE_integer("max_seq_length", 100, "")
flags.DEFINE_float("masked_lm_prob", 0.8, "Masked LM probability.")
# flags.DEFINE_integer("max_predictions_per_seq", 40, "")
flags.DEFINE_bool("do_train", True, "")
flags.DEFINE_bool("do_eval", False, "")
flags.DEFINE_integer("batch_size", 256, "")
flags.DEFINE_integer("epoch", 5, "")
flags.DEFINE_float("learning_rate", 1e-4, "")
flags.DEFINE_integer("num_train_steps", 1000000, "Number of training steps.")
flags.DEFINE_integer("num_warmup_steps", 100, "Number of warmup steps.")
flags.DEFINE_integer("save_checkpoints_steps", 8000, "")
flags.DEFINE_integer("iterations_per_loop", 2000, "How many steps to make in each estimator call.")
flags.DEFINE_integer("max_eval_steps", 1000, "Maximum number of eval steps.")
flags.DEFINE_integer("neg_sample_num", None, "The number of negative samples in a batch")
flags.DEFINE_string("neg_strategy", None, "Strategy of negative sampling")
flags.DEFINE_bool("neg_share", False, "Whether to share negative samples for each training batch")
flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")
flags.DEFINE_string("data_dir", '../inter_data/', "data dir.")
flags.DEFINE_string("bizdate", None, "the date of running experiments")

MAX_PREDICTIONS_PER_SEQ = math.ceil(FLAGS.max_seq_length * FLAGS.masked_lm_prob)

print("MAX_SEQUENCE_LENGTH:", FLAGS.max_seq_length)
print("MAX_PREDICTIONS_PER_SEQ:", MAX_PREDICTIONS_PER_SEQ)

if FLAGS.bizdate is None:
    raise ValueError("bizdate is required..")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool("use_pop_random", True, "use pop random negative samples")
flags.DEFINE_string("vocab_filename", "vocab", "vocab filename")


def input_fn(input_files,
             is_training,
             num_cpu_threads=4):
    """ The actual input function"""

    name_to_features = {
        "address":
            tf.FixedLenFeature([1], tf.int64),
        "input_ids":
            tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "input_positions":
            tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "input_counts":
            tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "input_mask":
            tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "input_io_flags":
            tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "input_values":
            tf.FixedLenFeature([FLAGS.max_seq_length], tf.int64),
        "masked_lm_positions":
            tf.FixedLenFeature([MAX_PREDICTIONS_PER_SEQ], tf.int64),
        "masked_lm_ids":
            tf.FixedLenFeature([MAX_PREDICTIONS_PER_SEQ], tf.int64),
        "masked_lm_weights":
            tf.FixedLenFeature([MAX_PREDICTIONS_PER_SEQ], tf.float32)
    }

    if is_training:
        d = tf.data.TFRecordDataset(input_files)
        d = d.repeat(FLAGS.epoch).shuffle(100)

    else:
        d = tf.data.TFRecordDataset(input_files)

    d = d.map(lambda record: _decode_record(record, name_to_features), num_parallel_calls=num_cpu_threads)
    d = d.batch(batch_size=FLAGS.batch_size)

    iterator = d.make_one_shot_iterator()
    features = iterator.get_next()

    return features


def model_fn(features, mode, bert_config, vocab, init_checkpoint, learning_rate,
             num_train_steps, num_warmup_steps, use_tpu, use_one_hot_embeddings):
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
        tf.logging.info("name = %s, shape = %s" % (name,
                                                   features[name].shape))

    input_ids = features["input_ids"]
    input_positions = features["input_positions"]
    input_mask = features["input_mask"]
    input_io_flags = features["input_io_flags"]
    input_values = features["input_values"]
    input_counts = features["input_counts"]
    masked_lm_positions = features["masked_lm_positions"]
    masked_lm_ids = features["masked_lm_ids"]
    masked_lm_weights = features["masked_lm_weights"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_positions=input_positions,
        input_io_flags=input_io_flags,
        input_amounts=input_values,
        input_counts=input_counts,
        input_mask=input_mask,
        token_type_ids=None,
        use_one_hot_embeddings=use_one_hot_embeddings)

    #         all_user_and_item = model.get_embedding_table()
    #         item_ids = [i for i in range(0, item_size + 1)]
    #         softmax_output_embedding = tf.nn.embedding_lookup(all_user_and_item, item_ids)

    if FLAGS.neg_share:
        print("-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+")
        print("use intra-batch negative sharing")
        print("neg-to-pos ratio=" + str(FLAGS.neg_sample_num) + ":1")
        pretrain_loss_calculation = get_masked_lm_output_negative_sampling

    else:
        print("-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+")
        print("not use intra-batch negative sharing")
        print("neg-to-pos ratio=" + str(FLAGS.neg_sample_num) + ":1")
        pretrain_loss_calculation = get_masked_lm_output_negative_sampling_unshare


    (masked_lm_loss,
     masked_lm_example_loss, masked_lm_log_probs) = pretrain_loss_calculation(
        bert_config,
        model.get_sequence_output(),
        model.get_embedding_table(),
        masked_lm_positions,
        masked_lm_ids,
        masked_lm_weights,
        vocab)  # model use the token embedding table as the output_weights

    total_loss = masked_lm_loss
    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None

    if init_checkpoint:
        (assignment_map, initialized_variable_names
         ) = modeling.get_assignment_map_from_checkpoint(
            tvars, init_checkpoint)
        if use_tpu:

            def tpu_scaffold():
                tf.train.init_from_checkpoint(init_checkpoint,
                                              assignment_map)
                return tf.train.Scaffold()

            scaffold_fn = tpu_scaffold
        else:
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
        init_string = ""
        if var.name in initialized_variable_names:
            init_string = ", *INIT_FROM_CKPT*"
        tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                        init_string)

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = optimization.create_optimizer(total_loss, learning_rate,
                                                 num_train_steps,
                                                 num_warmup_steps, use_tpu)

        return model, train_op, total_loss
        # output_spec = tf.estimator.EstimatorSpec(
        #     mode=mode,
        #     loss=total_loss,
        #     train_op=train_op,
        #     scaffold=scaffold_fn)

    elif mode == tf.estimator.ModeKeys.EVAL:

        def metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids, masked_lm_weights):
            """Computes the loss and accuracy of the model."""
            masked_lm_log_probs = tf.reshape(masked_lm_log_probs, [-1, masked_lm_log_probs.shape[-1]])
            masked_lm_predictions = tf.argmax(masked_lm_log_probs, axis=-1, output_type=tf.int32)
            masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
            masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
            masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
            masked_lm_accuracy = tf.metrics.accuracy(labels=masked_lm_ids, predictions=masked_lm_predictions,
                                                     weights=masked_lm_weights)
            masked_lm_mean_loss = tf.metrics.mean(values=masked_lm_example_loss, weights=masked_lm_weights)

            return {
                "masked_lm_accuracy": masked_lm_accuracy,
                "masked_lm_loss": masked_lm_mean_loss,
            }

        tf.add_to_collection('eval_sp', masked_lm_log_probs)
        tf.add_to_collection('eval_sp', input_ids)
        tf.add_to_collection('eval_sp', masked_lm_ids)

        eval_metrics = metric_fn(masked_lm_example_loss,
                                 masked_lm_log_probs,
                                 masked_lm_ids,
                                 masked_lm_weights)

        # output_spec = tf.estimator.EstimatorSpec(
        #     mode=mode,
        #     loss=total_loss,
        #     eval_metric_ops=eval_metrics,
        #     scaffold=scaffold_fn)

        return model, total_loss

    else:
        raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))


def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
    """Get loss and log probs for the masked LM."""
    # [batch_size*label_size, dim]
    input_tensor = gather_indexes(input_tensor, positions)

    with tf.variable_scope("cls/predictions"):
        # We apply one more non-linear transformation before the output layer.
        # This matrix is not used after pre-training.
        with tf.variable_scope("transform"):
            input_tensor = tf.layers.dense(
                input_tensor,
                units=bert_config.hidden_size,
                activation=modeling.get_activation(bert_config.hidden_act),
                kernel_initializer=modeling.create_initializer(
                    bert_config.initializer_range))
            input_tensor = modeling.layer_norm(input_tensor)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        output_bias = tf.get_variable(
            "output_bias",
            shape=[output_weights.shape[0]],
            initializer=tf.zeros_initializer())
        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        # logits, (bs*label_size, vocab_size)
        log_probs = tf.nn.log_softmax(logits, -1)

        label_ids = tf.reshape(label_ids, [-1])
        label_weights = tf.reshape(label_weights, [-1])

        one_hot_labels = tf.one_hot(
            label_ids, depth=output_weights.shape[0], dtype=tf.float32)

        # The `positions` tensor might be zero-padded (if the sequence is too
        # short to have the maximum number of predictions). The `label_weights`
        # tensor has a value of 1.0 for every real prediction and 0.0 for the
        # padding predictions.
        per_example_loss = -tf.reduce_sum(
            log_probs * one_hot_labels, axis=[-1])
        numerator = tf.reduce_sum(label_weights * per_example_loss)
        denominator = tf.reduce_sum(label_weights) + 1e-5
        loss = numerator / denominator

    return (loss, per_example_loss, log_probs)


def get_masked_lm_output_negative_sampling(bert_config, input_tensor, output_weights, positions,
                                           label_ids, label_weights, vocab):
    """Get loss and log probs for the masked LM."""

    # negative sample randomly
    word_num = len(vocab.vocab_words) - 3

    if FLAGS.neg_strategy == "uniform":
        neg_ids, _, _ = tf.nn.uniform_candidate_sampler(true_classes=[[len(vocab.vocab_words)]],
                                                        num_true=1,
                                                        num_sampled=FLAGS.neg_sample_num,
                                                        unique=True,
                                                        range_max=word_num)

    elif FLAGS.neg_strategy == "zip":
        neg_ids, _, _ = tf.nn.log_uniform_candidate_sampler(true_classes=[[len(vocab.vocab_words)]],
                                                            num_true=1,
                                                            num_sampled=FLAGS.neg_sample_num,
                                                            unique=True,
                                                            range_max=word_num)

    elif FLAGS.neg_strategy == "freq":
        # negative sample based on frequency
        neg_ids, _, _ = tf.nn.fixed_unigram_candidate_sampler(true_classes=[[len(vocab.vocab_words)]],
                                                              num_true=1,
                                                              num_sampled=FLAGS.neg_sample_num,
                                                              unique=True,
                                                              range_max=word_num,
                                                              unigrams=list(
                                                                  map(lambda x: pow(x, 1 / 2), vocab.frequency[3:]))
                                                              )

    else:
        raise ValueError("Please select correct negative sampling strategy: uniform, zip, .")

    neg_ids = neg_ids + 1 + 3 # (1 padding, 2 mask, 3 not use)

    # [batch_size*label_size, dim]
    input_tensor = gather_indexes(input_tensor, positions)

    with tf.variable_scope("cls/predictions"):
        # We apply one more non-linear transformation before the output layer.
        # This matrix is not used after pre-training.
        with tf.variable_scope("transform"):
            input_tensor = tf.layers.dense(
                input_tensor,
                units=bert_config.hidden_size,
                activation=modeling.get_activation(bert_config.hidden_act),
                kernel_initializer=modeling.create_initializer(
                    bert_config.initializer_range))
            input_tensor = modeling.layer_norm(input_tensor) # (batch_size * mask_num, dim)
        #
        label_ids = tf.reshape(label_ids, [-1]) # batch_size, mask_num
        label_weights = tf.reshape(label_weights, [-1]) # batch_size * mask_num
        pos_output_weights = tf.nn.embedding_lookup(output_weights, label_ids)  # (batch_size * mask_num, dim)
        neg_output_weights = tf.nn.embedding_lookup(output_weights, neg_ids)  # (sample_num, dim)

        pos_logits = tf.reduce_sum(tf.multiply(input_tensor, pos_output_weights), axis=-1)  # (batch_size * mask_num)
        pos_logits = tf.expand_dims(pos_logits, axis=1)
        neg_logits = tf.matmul(input_tensor, neg_output_weights, transpose_b=True)  # 768, 10000

        print("============")
        print(pos_logits)
        print(neg_logits)

        # raise ValueError("========Break==========")

        logits = tf.concat([pos_logits, neg_logits], axis=1)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        output_bias = tf.get_variable(
            "output_bias",
            shape=[logits.shape[1]],
            initializer=tf.zeros_initializer())

        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, -1)
        per_example_loss = -log_probs[:, 0]
        # The `positions` tensor might be zero-padded (if the sequence is too
        # short to have the maximum number of predictions). The `label_weights`
        # tensor has a value of 1.0 for every real prediction and 0.0 for the
        # padding predictions.
        numerator = tf.reduce_sum(label_weights * per_example_loss)
        denominator = tf.reduce_sum(label_weights) + 1e-5
        loss = numerator / denominator

    return (loss, per_example_loss, log_probs)



def get_masked_lm_output_negative_sampling_unshare(bert_config, input_tensor, output_weights, positions,
                                                   label_ids, label_weights, vocab):
    """Get loss and log probs for the masked LM."""

    # negative sample randomly
    word_num = len(vocab.vocab_words) - 3

    if FLAGS.neg_strategy == "uniform":
        neg_ids, _, _ = tf.nn.uniform_candidate_sampler(true_classes=[[len(vocab.vocab_words)]],
                                                        num_true=1,
                                                        num_sampled=FLAGS.neg_sample_num * FLAGS.batch_size,
                                                        unique=False,
                                                        range_max=word_num)


    elif FLAGS.neg_strategy == "zip":
        neg_ids, _, _ = tf.nn.log_uniform_candidate_sampler(true_classes=[[len(vocab.vocab_words)]],
                                                            num_true=1,
                                                            num_sampled=FLAGS.neg_sample_num * FLAGS.batch_size,
                                                            unique=False,
                                                            range_max=word_num)

    elif FLAGS.neg_strategy == "freq":
        # negative sample based on frequency
        neg_ids, _, _ = tf.nn.fixed_unigram_candidate_sampler(true_classes=[[len(vocab.vocab_words)]],
                                                              num_true=1,
                                                              num_sampled=FLAGS.neg_sample_num * FLAGS.batch_size,
                                                              unique=False,
                                                              range_max=word_num,
                                                              unigrams=list(
                                                                  map(lambda x: pow(x, 1 / 2), vocab.frequency[3:]))
                                                              )

    else:
        raise ValueError("Please select correct negative sampling strategy: uniform, zip, .")

    neg_ids = neg_ids + 1 + 3 # (1 padding, 2 mask, 3 not use)

    # [batch_size*label_size, dim]
    input_tensor = gather_indexes(input_tensor, positions)

    with tf.variable_scope("cls/predictions"):
        # We apply one more non-linear transformation before the output layer.
        # This matrix is not used after pre-training.
        with tf.variable_scope("transform"):
            input_tensor = tf.layers.dense(
                input_tensor,
                units=bert_config.hidden_size,
                activation=modeling.get_activation(bert_config.hidden_act),
                kernel_initializer=modeling.create_initializer(
                    bert_config.initializer_range))
            input_tensor = modeling.layer_norm(input_tensor) # (batch_size * mask_num, dim)

        input_tensor = tf.reshape(input_tensor, [-1, MAX_PREDICTIONS_PER_SEQ, bert_config.hidden_size])
        #
        label_ids = tf.reshape(label_ids, [-1]) # batch_size, mask_num
        label_weights = tf.reshape(label_weights, [-1]) # batch_size * mask_num
        pos_output_weights = tf.nn.embedding_lookup(output_weights, label_ids)  # (batch_size * mask_num, dim)
        pos_output_weights = tf.reshape(pos_output_weights, [-1, MAX_PREDICTIONS_PER_SEQ, bert_config.hidden_size])

        neg_output_weights = tf.nn.embedding_lookup(output_weights, neg_ids)  # (sample_num, dim)
        neg_output_weights = tf.reshape(neg_output_weights, [-1, FLAGS.neg_sample_num, bert_config.hidden_size])

        pos_logits = tf.reduce_sum(tf.multiply(input_tensor, pos_output_weights), axis=-1)  # (batch_size, mask_num)
        pos_logits = tf.expand_dims(pos_logits, axis=2) # (batch_size, mask_num, 1)

        neg_logits = tf.matmul(input_tensor, neg_output_weights, transpose_b=True)  # [batch_size, mask_num, neg_sample_num]

        print("============")
        print(pos_logits)
        print(neg_logits)

        # raise ValueError("========Break==========")

        logits = tf.concat([pos_logits, neg_logits], axis=2)
        logits = tf.reshape(logits, [-1, logits.shape[2]])
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        output_bias = tf.get_variable(
            "output_bias",
            shape=[logits.shape[1]],
            initializer=tf.zeros_initializer())

        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, -1)
        per_example_loss = -log_probs[:, 0]
        # The `positions` tensor might be zero-padded (if the sequence is too
        # short to have the maximum number of predictions). The `label_weights`
        # tensor has a value of 1.0 for every real prediction and 0.0 for the
        # padding predictions.
        numerator = tf.reduce_sum(label_weights * per_example_loss)
        denominator = tf.reduce_sum(label_weights) + 1e-5
        loss = numerator / denominator

    return (loss, per_example_loss, log_probs)

def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]
    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                      [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor


def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.to_int32(t)
        example[name] = t
    return example


def main(_):
    if FLAGS.do_train:
        mode = tf.estimator.ModeKeys.TRAIN
        input_files = FLAGS.train_input_file + "." + FLAGS.bizdate
        # load data
        features = input_fn(input_files, is_training=True)

    elif FLAGS.do_eval:
        mode = tf.estimator.ModeKeys.EVAL
        input_files = FLAGS.test_input_file + "." + FLAGS.bizdate
        features = input_fn(input_files, is_training=False)

    else:
        raise ValueError("Only TRAIN and EVAL modes are supported.")

    # modeling
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    tf.gfile.MakeDirs(FLAGS.checkpointDir)

    # load vocab
    vocab_file_name = FLAGS.data_dir + FLAGS.vocab_filename + "." + FLAGS.bizdate
    with open(vocab_file_name, "rb") as f:
        vocab = pkl.load(f)

    if FLAGS.do_train:
        bert_model, train_op, total_loss = model_fn(features, mode, bert_config, vocab, FLAGS.init_checkpoint,
                                                    FLAGS.learning_rate,
                                                    FLAGS.num_train_steps, FLAGS.num_warmup_steps, False, False)
        # saver define
        tvars = tf.trainable_variables()
        saver = tf.train.Saver(max_to_keep=30, var_list=tvars)

        # start session
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            losses = []
            iter = 0
            # start = time.time()
            start = timer()
            while True:
                try:
                    _, loss = sess.run([train_op, total_loss])
                    # loss = sess.run([total_loss])

                    losses.append(loss)

                    if iter % 100 == 0:
                        # end = time.time()
                        end = timer()
                        loss = np.mean(losses)
                        print("iter=%d, loss=%f, time=%.2fs" % (iter, loss, end - start))
                        losses = []
                        # start = time.time()
                        start = timer()

                    if iter % FLAGS.save_checkpoints_steps == 0 and iter > 0:
                        saver.save(sess, os.path.join(FLAGS.checkpointDir, "model_" + str(round(iter))))

                    iter += 1

                except Exception as e:
                    # print("Out of Sequence, end of training...")
                    print(e)
                    # save model
                    saver.save(sess, os.path.join(FLAGS.checkpointDir, "model_" + str(round(iter))))
                    break

    elif FLAGS.do_eval:
        # must have checkpoint
        if FLAGS.init_checkpoint == None:
            raise ValueError("Must need a checkpoint for evaluation")

        bert_model, total_loss = model_fn(features, mode, bert_config, vocab, FLAGS.init_checkpoint,
                                          FLAGS.learning_rate,
                                          FLAGS.num_train_steps, FLAGS.num_warmup_steps, False, False)

        # start session
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            losses = []
            iter = 0
            # start = time.time()
            start = timer()
            while True:
                try:
                    loss = sess.run(total_loss)
                    losses.append(loss)

                    if iter % 500 == 0:
                        # end = time.time()
                        end = timer()
                        print("iter=%d, time=%.2fs" % (iter, end - start))
                        # start = time.time()
                        start = timer()

                    iter += 1

                except Exception as e:
                    print("Out of Sequence")
                    # save model
                    # saver.save(sess, os.path.join(FLAGS.checkpointDir, "model_" + str(iter)))
                    break

            final_loss = np.mean(losses)
            eval_sample_num = len(losses)

            print("========Evaluation Results==========")
            print("sample_num=%d, loss=%.2f" % (eval_sample_num, final_loss))

    else:
        raise ValueError("Only TRAIN and EVAL modes are supported.")

    return


if __name__ == '__main__':
    tf.app.run()