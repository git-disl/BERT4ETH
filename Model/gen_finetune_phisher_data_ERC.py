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
flags.DEFINE_bool("random_drop", False, "whether to drop randomly")
flags.DEFINE_bool("total_drop", False, "whether to drop")
flags.DEFINE_bool("drop", False, "whether to drop")
flags.DEFINE_float("beta", 0.0, "penalty for drop")
flags.DEFINE_float("drop_ratio", 0.0, "ratio to drop repetitive and frequent trans")

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

    def __init__(self, address, tokens, label):

        self.address = [address]
        self.tokens = list(map(lambda x: x[0], tokens))
        self.block_timestamps = list(map(lambda x: x[2], tokens))
        self.values = list(map(lambda x: x[3], tokens))
        self.label = label

        def map_io_flag(token):
            flag = token[4]
            if flag == "OUT":
                return 1
            elif flag == "IN":
                return 2
            else:
                return 0

        def map_trans_flag(token):
            flag = token[6]
            if flag == "TRANS":
                return 1
            elif flag == "ERC20":
                return 2
            else:
                return 0

        self.io_flags = list(map(map_io_flag, tokens))
        self.trans_flags = list(map(map_trans_flag, tokens))
        self.cnts = list(map(lambda x: x[5], tokens))
        self.erc20_nums = list(map(lambda x: x[7], tokens))


    def __str__(self):
        s = "address: %s\n" % (self.address[0])
        s += "tokens: %s\n" % (
            " ".join([printable_text(x) for x in self.tokens]))
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

        tokens = sequences[i]
        address = tokens[0][0]

        instance = FinetuneInstance(
            address=address,
            tokens=tokens,
            label=label_list[i])
        instances.append(instance)

    end = time.time()
    print("=======Finish========")
    print("cost time:%.2f" % (end - start))
    return instances


def create_bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
    return feature

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
        input_ids = vocab.convert_tokens_to_ids(instance.tokens)
        address = vocab.convert_tokens_to_ids(instance.address)
        counts = instance.cnts
        block_timestamps = instance.block_timestamps
        values = instance.cnts
        io_flags = instance.io_flags
        trans_flags = instance.trans_flags
        erc20_nums = instance.erc20_nums # the number of erc20log eoa address
        positions = convert_timestamp_to_position(block_timestamps)
        label = [instance.label]

        input_mask = [1] * len(input_ids)
        assert len(input_ids) <= max_seq_length
        assert len(counts) <= max_seq_length
        assert len(values) <= max_seq_length
        assert len(io_flags) <= max_seq_length
        assert len(trans_flags) <= max_seq_length
        assert len(erc20_nums) <= max_seq_length
        assert len(positions) <= max_seq_length

        input_ids += ["0"] * (max_seq_length - len(input_ids))
        counts += [0] * (max_seq_length - len(counts))
        values += [0] * (max_seq_length - len(values))
        io_flags += [0] * (max_seq_length - len(io_flags))
        trans_flags += [0] * (max_seq_length - len(trans_flags))
        erc20_nums += [0] * (max_seq_length - len(erc20_nums))
        positions += [0] * (max_seq_length - len(positions))
        input_mask += [0] * (max_seq_length - len(input_mask))

        assert len(input_ids) == max_seq_length
        assert len(counts) == max_seq_length
        assert len(values) == max_seq_length
        assert len(io_flags) == max_seq_length
        assert len(positions) == max_seq_length
        assert len(input_mask) == max_seq_length

        features = collections.OrderedDict()
        features["address"] = create_int_feature(address)
        features["label"] = create_float_feature(label)
        features["input_ids"] = create_bytes_feature(list(map(lambda x: str(x).encode("utf-8"), input_ids)))
        features["input_positions"] = create_int_feature(positions)
        features["input_counts"] = create_int_feature(counts)
        features["input_io_flags"] = create_int_feature(io_flags)
        features["input_trans_flags"] = create_int_feature(trans_flags)
        features["input_erc20_nums"] = create_int_feature(erc20_nums)
        features["input_values"] = create_int_feature(values)
        features["input_mask"] = create_int_feature(input_mask)

        tf_example = tf.train.Example(
            features=tf.train.Features(feature=features))

        writers[writer_index].write(tf_example.SerializeToString())
        writer_index = (writer_index + 1) % len(writers)

        total_written += 1

        if inst_index < 3:
            tf.logging.info("*** Example ***")
            tf.logging.info("tokens: %s" % " ".join(
                [printable_text(x) for x in instance.tokens]))

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


def repeat_drop(eoa2seq, vocab, beta, ratio):
    """
    ratio: the ratio of reduce transaction
    beta: the penalty for sampling, if beta is too huge, there is overflow
    """
    new_eoa2seq = {}

    for eoa, seq in eoa2seq.items():

        filter_num = int(ratio * len(seq))
        # calculate drop frequency after each drop
        address_list = list(map(lambda x: x[0], seq))
        freq = {}
        for addr in address_list:
            try:
                freq[addr] += 1
            except:
                freq[addr] = 1

        frequency_list = [freq[addr] for addr in address_list]
        if np.max(frequency_list) == 1:
            # if there is no repetitiveness in the sequence
            new_eoa2seq[eoa] = seq
            continue

        frequency_list = list(map(lambda x: pow(x, beta), frequency_list))
        denominator = np.sum(frequency_list)
        proba_list = frequency_list / denominator
        drop_idx = set(np.random.choice(range(len(seq)), filter_num, p=proba_list, replace=False))

        new_seq = []
        for i in range(len(seq)):
            if i not in drop_idx:
                new_seq.append(seq[i])

        new_eoa2seq[eoa] = new_seq

    return new_eoa2seq

def total_repeat_drop(eoa2seq):
    """
    totally drop the repeat part.
    """
    new_eoa2seq = {}
    for eoa, seq in eoa2seq.items():
        new_seq = []
        exist_addr = set()
        for trans in seq:
            if trans[0] not in exist_addr:
                exist_addr.add(trans[0])
                new_seq.append(trans)

        new_eoa2seq[eoa] = new_seq

    return new_eoa2seq


def random_drop(eoa2seq, ratio=0.5):

    new_eoa2seq = {}
    for eoa, seq in eoa2seq.items():
        filter_num = int(ratio * len(seq))

        if len(seq) <= 2:
            new_eoa2seq[eoa] = seq

        else:
            remain_idx = set(np.random.choice(range(len(seq)), len(seq) - filter_num, replace=False))
            new_seq = []

            for id in remain_idx:
                new_seq.append(seq[id])

            new_seq = sorted(new_seq, key=functools.cmp_to_key(cmp_udf_reverse))
            new_eoa2seq[eoa] = new_seq

    return new_eoa2seq



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

    if FLAGS.random_drop:
        eoa2seq = random_drop(eoa2seq, FLAGS.drop_ratio)

    elif FLAGS.drop:
        eoa2seq = repeat_drop(eoa2seq, None, FLAGS.beta, FLAGS.drop_ratio)

    elif FLAGS.total_drop:
        eoa2seq = total_repeat_drop(eoa2seq)

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
            seqs.append([[eoa, 0, 0, 0, 0, 0, 0, 0]])
            seqs[idx] += seq
            idx += 1
            label_list.append(is_phish(eoa))

        elif len(seq) > max_num_tokens:
            beg_idx = list(range(len(seq) - max_num_tokens, 0, -1 * SLIDING_STEP))
            beg_idx.append(0)

            if len(beg_idx) > 500:
                beg_idx = list(np.random.permutation(beg_idx)[:500])
                for i in beg_idx:
                    seqs.append([[eoa, 0, 0, 0, 0, 0, 0, 0]])
                    seqs[idx] += seq[i:i + max_num_tokens]
                    idx += 1
                    label_list.append(is_phish(eoa))

            else:
                for i in beg_idx[::-1]:
                    seqs.append([[eoa, 0, 0, 0, 0, 0, 0, 0]])
                    seqs[idx] += seq[i:i + max_num_tokens]
                    idx += 1
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
        if seq[0][0] in train_eoa_list:
            train_seqs.append(seq)
            train_label_list.append(label)
        elif seq[0][0] in test_eoa_list:
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
