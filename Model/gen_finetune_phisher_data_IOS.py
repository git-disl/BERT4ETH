from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
from vocab import FreqVocab
import os
import collections
import random
import functools
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import sys
import six
import pickle as pkl
import time

flags = tf.flags
FLAGS = flags.FLAGS

random_seed = 12345
rng = random.Random(random_seed)

## parameters
flags.DEFINE_integer("max_seq_length", 50, "max sequence length.")
# flags.DEFINE_integer("sliding_step", 30, "sliding window step size.")
flags.DEFINE_string("data_dir", './data/', "data dir.")
flags.DEFINE_string("dataset_name", 'eth',"dataset name.")
flags.DEFINE_string("vocab_filename", "vocab", "vocab filename")
flags.DEFINE_string("bizdate", None, "the signature of running experiments")
flags.DEFINE_string("source_bizdate", None, "signature of previous data")

SLIDING_STEP = round(FLAGS.max_seq_length * 0.6)

print("MAX_SEQUENCE_LENGTH:", FLAGS.max_seq_length)
print("SLIDING_STEP:", SLIDING_STEP)

if FLAGS.bizdate is None:
    raise ValueError("bizdate is required.")

if FLAGS.source_bizdate is None:
    raise ValueError("source_bizdate is required.")

class FinetuneInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, address, in_tokens, out_tokens, all_tokens, label):

        def map_io_flag(token):
            flag = token[4]
            if flag == "OUT":
                return 1
            elif flag == "IN":
                return 2
            else:
                return 0

        self.address = [address]
        self.label = label

        # in
        self.in_tokens = list(map(lambda x: x[0], in_tokens))
        self.in_io_flags = list(map(map_io_flag, in_tokens))
        self.in_block_timestamps = list(map(lambda x: x[2], in_tokens))
        self.in_values = list(map(lambda x: x[3], in_tokens))
        self.in_cnts = list(map(lambda x: x[5], in_tokens))

        # out
        self.out_tokens = list(map(lambda x: x[0], out_tokens))
        self.out_io_flags = list(map(map_io_flag, out_tokens))
        self.out_block_timestamps = list(map(lambda x: x[2], out_tokens))
        self.out_values = list(map(lambda x: x[3], out_tokens))
        self.out_cnts = list(map(lambda x: x[5], out_tokens))

        # all
        self.all_tokens = list(map(lambda x: x[0], all_tokens))
        self.all_io_flags = list(map(map_io_flag, all_tokens))
        self.all_block_timestamps = list(map(lambda x: x[2], all_tokens))
        self.all_values = list(map(lambda x: x[3], all_tokens))
        self.all_cnts = list(map(lambda x: x[5], all_tokens))



    def __str__(self):
        s = "address: %s\n" % (self.address[0])
        s += "in_tokens: %s\n" % (
            " ".join([printable_text(x) for x in self.in_tokens]))

        s += "out_tokens: %s\n" % (
            " ".join([printable_text(x) for x in self.out_tokens]))
        s += "\n"

        s += "all_tokens: %s\n" % (
            " ".join([printable_text(x) for x in self.all_tokens]))
        s += "\n"

        return s


    def __repr__(self):
        return self.__str__()

def printable_text(text):
    """Returns text encoded in a way suitable for print or `tf.logging`."""

    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text
        elif isinstance(text, unicode):
            return text.encode("utf-8")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")

def cmp_udf_reverse(x1, x2):
    time1 = int(x1[2])
    time2 = int(x2[2])

    if time1 < time2:
        return 1
    elif time1 > time2:
        return -1
    else:
        return 0


def create_embedding_predictions(tokens):
    """Creates the predictions for the masked LM objective."""
    address = tokens[0][0]
    output_tokens = tokens
    masked_lm_positions = []
    masked_lm_labels = []
    return (address, output_tokens, masked_lm_positions, masked_lm_labels)


def gen_finetune_samples(sequences, label_list):
    instances = []
    # create train
    start = time.time()
    for i in tqdm(range(len(sequences))):

        in_tokens, out_tokens, all_tokens = sequences[i]
        address = all_tokens[0][0]

        instance = FinetuneInstance(
            address=address,
            in_tokens=in_tokens,
            out_tokens=out_tokens,
            all_tokens=all_tokens,
            label=label_list[i])
        instances.append(instance)

    end = time.time()
    print("=======Finish========")
    print("cost time:%.2f" % (end - start))
    return instances

def create_int_feature(values):
    feature = tf.train.Feature(
        int64_list=tf.train.Int64List(value=list(values)))
    return feature

def create_float_feature(values):
    feature = tf.train.Feature(
        float_list=tf.train.FloatList(value=list(values)))
    return feature

def convert_timestamp_to_position(block_timestamps):
    position = [0]
    if len(block_timestamps) <= 1:
        return position
    last_ts = block_timestamps[1]
    idx = 1
    for b_ts in block_timestamps[1:]:
        if b_ts != last_ts:
            last_ts = b_ts
            idx += 1
        position.append(idx)
    return position

def write_finetune_instance_to_example_files(instances, max_seq_length, vocab, output_files):
    """Create TF example files from `TrainingInstance`s."""
    writers = []
    for output_file in output_files:
        writers.append(tf.python_io.TFRecordWriter(output_file))

    writer_index = 0
    total_written = 0

    for inst_index in tqdm(range(len(instances))):
        instance = instances[inst_index]
        address = vocab.convert_tokens_to_ids(instance.address)
        label = [instance.label]

        # ------------------- in sequence ---------------------
        in_token_ids = vocab.convert_tokens_to_ids(instance.in_tokens)
        in_counts = instance.in_cnts
        in_io_flags = instance.in_io_flags
        in_block_timestamps = instance.in_block_timestamps
        in_values = instance.in_values
        in_positions = convert_timestamp_to_position(in_block_timestamps)

        in_mask = [1] * len(in_token_ids)
        assert len(in_token_ids) <= max_seq_length
        assert len(in_counts) <= max_seq_length
        assert len(in_values) <= max_seq_length
        assert len(in_io_flags) <= max_seq_length
        assert len(in_positions) <= max_seq_length

        in_token_ids += [0] * (max_seq_length - len(in_token_ids))
        in_counts += [0] * (max_seq_length - len(in_counts))
        in_values += [0] * (max_seq_length - len(in_values))
        in_io_flags += [0] * (max_seq_length - len(in_io_flags))
        in_positions += [0] * (max_seq_length - len(in_positions))
        in_mask += [0] * (max_seq_length - len(in_mask))

        assert len(in_token_ids) == max_seq_length
        assert len(in_counts) == max_seq_length
        assert len(in_values) == max_seq_length
        assert len(in_io_flags) == max_seq_length
        assert len(in_positions) == max_seq_length
        assert len(in_mask) == max_seq_length

        # ------------------- out sequence ---------------------
        out_token_ids = vocab.convert_tokens_to_ids(instance.out_tokens)
        out_counts = instance.out_cnts
        out_io_flags = instance.out_io_flags
        out_block_timestamps = instance.out_block_timestamps
        out_values = instance.out_values
        out_positions = convert_timestamp_to_position(out_block_timestamps)

        out_mask = [1] * len(out_token_ids)
        assert len(out_token_ids) <= max_seq_length
        assert len(out_counts) <= max_seq_length
        assert len(out_values) <= max_seq_length
        assert len(out_io_flags) <= max_seq_length
        assert len(out_positions) <= max_seq_length

        out_token_ids += [0] * (max_seq_length - len(out_token_ids))
        out_counts += [0] * (max_seq_length - len(out_counts))
        out_values += [0] * (max_seq_length - len(out_values))
        out_io_flags += [0] * (max_seq_length - len(out_io_flags))
        out_positions += [0] * (max_seq_length - len(out_positions))
        out_mask += [0] * (max_seq_length - len(out_mask))

        assert len(out_token_ids) == max_seq_length
        assert len(out_counts) == max_seq_length
        assert len(out_values) == max_seq_length
        assert len(out_io_flags) == max_seq_length
        assert len(out_positions) == max_seq_length
        assert len(out_mask) == max_seq_length

        # -------------- all sequence ------------------
        all_token_ids = vocab.convert_tokens_to_ids(instance.all_tokens)
        all_counts = instance.all_cnts
        all_io_flags = instance.all_io_flags
        all_block_timestamps = instance.all_block_timestamps
        all_values = instance.all_cnts
        all_positions = convert_timestamp_to_position(all_block_timestamps)

        all_mask = [1] * len(all_token_ids)
        assert len(all_token_ids) <= max_seq_length
        assert len(all_io_flags) <= max_seq_length
        assert len(all_counts) <= max_seq_length
        assert len(all_values) <= max_seq_length
        assert len(all_positions) <= max_seq_length

        all_token_ids += [0] * (max_seq_length - len(all_token_ids))
        all_io_flags += [0] * (max_seq_length - len(all_io_flags))
        all_counts += [0] * (max_seq_length - len(all_counts))
        all_values += [0] * (max_seq_length - len(all_values))
        all_positions += [0] * (max_seq_length - len(all_positions))
        all_mask += [0] * (max_seq_length - len(all_mask))

        assert len(all_token_ids) == max_seq_length
        assert len(all_io_flags) == max_seq_length
        assert len(all_counts) == max_seq_length
        assert len(all_values) == max_seq_length
        assert len(all_positions) == max_seq_length
        assert len(all_mask) == max_seq_length

        # feature generation
        features = collections.OrderedDict()
        features["address"] = create_int_feature(address)
        features["label"] = create_float_feature(label)

        # in sequence
        features["in_token_ids"] = create_int_feature(in_token_ids)
        features["in_positions"] = create_int_feature(in_positions)
        features["in_counts"] = create_int_feature(in_counts)
        features["in_io_flags"] = create_int_feature(in_io_flags)
        features["in_values"] = create_int_feature(in_values)
        features["in_mask"] = create_int_feature(in_mask)

        # out sequence
        features["out_token_ids"] = create_int_feature(out_token_ids)
        features["out_positions"] = create_int_feature(out_positions)
        features["out_counts"] = create_int_feature(out_counts)
        features["out_io_flags"] = create_int_feature(out_io_flags)
        features["out_values"] = create_int_feature(out_values)
        features["out_mask"] = create_int_feature(out_mask)

        # all sequence
        features["all_token_ids"] = create_int_feature(all_token_ids)
        features["all_positions"] = create_int_feature(all_positions)
        features["all_io_flags"] = create_int_feature(all_io_flags)
        features["all_counts"] = create_int_feature(all_counts)
        features["all_values"] = create_int_feature(all_values)
        features["all_mask"] = create_int_feature(all_mask)

        tf_example = tf.train.Example(
            features=tf.train.Features(feature=features))

        writers[writer_index].write(tf_example.SerializeToString())
        writer_index = (writer_index + 1) % len(writers)

        total_written += 1

        if inst_index < 3:
            tf.logging.info("*** Example ***")
            tf.logging.info("in_tokens: %s" % " ".join(
                [printable_text(x) for x in instance.in_tokens]))
            tf.logging.info("out_tokens: %s" % " ".join(
                [printable_text(x) for x in instance.out_tokens]))
            tf.logging.info("all_tokens: %s" % " ".join(
                [printable_text(x) for x in instance.all_tokens]))

            for feature_name in features.keys():
                feature = features[feature_name]
                values = []
                if feature.int64_list.value:
                    values = feature.int64_list.value
                elif feature.float_list.value:
                    values = feature.float_list.value
                tf.logging.info("%s: %s" % (feature_name,
                                            " ".join([str(x)
                                                      for x in values])))

    for writer in writers:
        writer.close()

    tf.logging.info("Wrote %d total instances", total_written)



if __name__ == '__main__':

    # load label
    phisher_account = pd.read_csv("../Data/phisher_account.txt", names=["account"])
    phisher_account_set = set(phisher_account.account.values)

    # load vocab
    vocab_file_name = FLAGS.data_dir + FLAGS.vocab_filename + "." + FLAGS.bizdate
    with open(vocab_file_name, "rb") as f:
        vocab = pkl.load(f)

    with open("./data/eoa2seq_" + FLAGS.source_bizdate + ".pkl", "rb") as f:
        eoa2seq = pkl.load(f)

    print("number of target user account:", len(eoa2seq))

    eoa_list = list(eoa2seq.keys())
    rng.shuffle(eoa_list)
    idx = round(len(eoa_list) * 0.7)
    train_eoa_list = set(eoa_list[:idx])
    test_eoa_list = set(eoa_list[idx:])
    print("------------------")
    print(len(train_eoa_list.intersection(test_eoa_list)))

    label_list = []
    # clip and add label
    def is_phish(address):
        if address in phisher_account_set:
            return 1.0
        else:
            return 0.0

    max_num_tokens = FLAGS.max_seq_length - 1
    seqs = []
    idx = 0

    for eoa, seq in eoa2seq.items():

        if len(seq) <= max_num_tokens:
            seq_in = []
            seq_in.append([eoa, 0, 0, 0, 0, 0])
            seq_out = []
            seq_out.append([eoa, 0, 0, 0, 0, 0])
            seq_all = []
            seq_all.append([eoa, 0, 0, 0, 0, 0])

            for trans in seq:
                if trans[4] == "IN":
                    seq_in.append(trans)
                elif trans[4] == "OUT":
                    seq_out.append(trans)
                else:
                    raise Exception("WRONG IN/OUT FLAG!")
                seq_all.append(trans)

            seqs.append([seq_in, seq_out, seq_all])
            label_list.append(is_phish(eoa))

        elif len(seq) > max_num_tokens:
            beg_idx = list(range(len(seq) - max_num_tokens, 0, -1 * SLIDING_STEP))
            beg_idx.append(0)

            if len(beg_idx) > 500:
                beg_idx = list(np.random.permutation(beg_idx)[:500])

            for i in beg_idx:
                seq_in = []
                seq_in.append([eoa, 0, 0, 0, 0, 0])
                seq_out = []
                seq_out.append([eoa, 0, 0, 0, 0, 0])
                seq_all = []
                seq_all.append([eoa, 0, 0, 0, 0, 0])

                for trans in seq[i:i + max_num_tokens]:
                    if trans[4] == "IN":
                        seq_in.append(trans)
                    elif trans[4] == "OUT":
                        seq_out.append(trans)
                    else:
                        raise Exception("WRONG IN/OUT FLAG!")
                    seq_all.append(trans)

                seqs.append([seq_in, seq_out, seq_all])
                label_list.append(is_phish(eoa))

    # split into training and testing sequences
    train_seqs = []
    test_seqs = []
    train_label_list = []
    test_label_list = []
    print("Splitting the sequence..")
    for i in tqdm(range(len(seqs))):
        seq = seqs[i]
        label = label_list[i]
        if seq[0][0][0] in train_eoa_list:
            train_seqs.append(seq)
            train_label_list.append(label)
        elif seq[0][0][0] in test_eoa_list:
            test_seqs.append(seq)
            test_label_list.append(label)

    print("Generating training samples..")
    train_phish_instance = gen_finetune_samples(train_seqs, train_label_list)
    rng.shuffle(train_phish_instance)

    print("Generating testing samples..")
    test_phish_instance = gen_finetune_samples(test_seqs, test_label_list)
    rng.shuffle(test_phish_instance)

    print("*** Writing to output files ***")
    output_filename = FLAGS.data_dir + "finetune_train.tfrecord" + "." + FLAGS.bizdate
    print("  %s", output_filename)

    write_finetune_instance_to_example_files(train_phish_instance, FLAGS.max_seq_length, vocab, [output_filename])

    print("*** Writing to output files ***")
    output_filename = FLAGS.data_dir + "finetune_test.tfrecord" + "." + FLAGS.bizdate
    print("  %s", output_filename)

    write_finetune_instance_to_example_files(test_phish_instance, FLAGS.max_seq_length, vocab, [output_filename])
    print("Finished..")
