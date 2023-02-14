import pandas as pd
import numpy as np
import pickle as pkl
from tqdm import tqdm
import collections
import functools
import random
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import six
import time
import math
from vocab import FreqVocab

tf.logging.set_verbosity(tf.logging.INFO)

random_seed = 12345
rng = random.Random(random_seed)

short_seq_prob = 0  # Probability of creating sequences which are shorter than the maximum length。
flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer("pool_size", 10, "multiprocesses pool size.")
flags.DEFINE_integer("max_seq_length", 50, "max sequence length.")
# flags.DEFINE_integer("max_predictions_per_seq", 40, "max_predictions_per_seq.")
flags.DEFINE_float("masked_lm_prob", 0.8, "Masked LM probability.")
flags.DEFINE_float("mask_prob", 1.0, "mask probabaility")
flags.DEFINE_bool("do_eval", False, "")
flags.DEFINE_bool("do_embed", True, "")
flags.DEFINE_integer("dupe_factor", 10, "Number of times to duplicate the input data (with different masks).")
# flags.DEFINE_integer("sliding_step", 30, "sliding window step size.")
flags.DEFINE_string("data_dir", './data/', "data dir.")
flags.DEFINE_string("vocab_filename", "vocab", "vocab filename")
flags.DEFINE_bool("total_drop", False, "whether to drop")
flags.DEFINE_bool("total_drop_random", False, "whether to drop totally and randomly")
flags.DEFINE_bool("drop", False, "whether to drop")
flags.DEFINE_bool("iter_drop", False, "whether to drop")
flags.DEFINE_float("beta", 0.0, "penalty for drop")
flags.DEFINE_float("drop_ratio", 0.5, "ratio to drop repetitive and frequent trans")
flags.DEFINE_bool("random_drop", False, "wether to drop randomly")

flags.DEFINE_string("bizdate", None, "the signature of running experiments")
flags.DEFINE_string("source_bizdate", None, "signature of previous data")

if FLAGS.bizdate is None:
    raise ValueError("bizdate is required.")

if FLAGS.source_bizdate is None:
    raise ValueError("source_bizdate is required.")

HEADER = 'hash,nonce,block_hash,block_number,transaction_index,from_address,to_address,value,gas,gas_price,input,block_timestamp,max_fee_per_gas,max_priority_fee_per_gas,transaction_type'.split(
    ",")

MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])

MAX_PREDICTIONS_PER_SEQ = math.ceil(FLAGS.max_seq_length * FLAGS.masked_lm_prob)
SLIDING_STEP = round(FLAGS.max_seq_length * 0.6)

print("MAX_SEQUENCE_LENGTH:", FLAGS.max_seq_length)
print("MAX_PREDICTIONS_PER_SEQ:", MAX_PREDICTIONS_PER_SEQ)
print("SLIDING_STEP:", SLIDING_STEP)

class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, address,
                 in_tokens, in_masked_lm_positions, in_masked_lm_labels,
                 out_tokens, out_masked_lm_positions, out_masked_lm_labels,
                 all_tokens, all_masked_lm_positions, all_masked_lm_labels):


        def map_io_flag(token):
            flag = token[4]
            if flag == "OUT":
                return 1
            elif flag == "IN":
                return 2
            else:
                return 0

        self.address = [address]
        # in
        self.in_tokens = list(map(lambda x: x[0], in_tokens))
        self.in_io_flags = list(map(map_io_flag, in_tokens))
        self.in_block_timestamps = list(map(lambda x: x[2], in_tokens))
        self.in_values = list(map(lambda x: x[3], in_tokens))
        self.in_cnts = list(map(lambda x: x[5], in_tokens))
        self.in_masked_lm_positions = in_masked_lm_positions
        self.in_masked_lm_labels = in_masked_lm_labels

        # out
        self.out_tokens = list(map(lambda x: x[0], out_tokens))
        self.out_io_flags = list(map(map_io_flag, out_tokens))
        self.out_block_timestamps = list(map(lambda x: x[2], out_tokens))
        self.out_values = list(map(lambda x: x[3], out_tokens))
        self.out_cnts = list(map(lambda x: x[5], out_tokens))
        self.out_masked_lm_positions = out_masked_lm_positions
        self.out_masked_lm_labels = out_masked_lm_labels

        # all
        self.all_tokens = list(map(lambda x: x[0], all_tokens))
        self.all_io_flags = list(map(map_io_flag, all_tokens))
        self.all_block_timestamps = list(map(lambda x: x[2], all_tokens))
        self.all_values = list(map(lambda x: x[3], all_tokens))
        self.all_cnts = list(map(lambda x: x[5], all_tokens))
        self.all_masked_lm_positions = all_masked_lm_positions
        self.all_masked_lm_labels = all_masked_lm_labels

    def __str__(self):
        s = "address: %s\n" % (self.address[0])
        s += "tokens: %s\n" % (
            " ".join([printable_text(x) for x in self.in_tokens]))
        s += "masked_lm_positions: %s\n" % (
            " ".join([str(x) for x in self.in_masked_lm_positions]))
        s += "masked_lm_labels: %s\n" % (
            " ".join([printable_text(x) for x in self.in_masked_lm_labels]))

        s += "tokens: %s\n" % (
            " ".join([printable_text(x) for x in self.out_tokens]))
        s += "masked_lm_positions: %s\n" % (
            " ".join([str(x) for x in self.out_masked_lm_positions]))
        s += "masked_lm_labels: %s\n" % (
            " ".join([printable_text(x) for x in self.out_masked_lm_labels]))
        s += "\n"

        s += "tokens: %s\n" % (
            " ".join([printable_text(x) for x in self.all_tokens]))
        s += "masked_lm_positions: %s\n" % (
            " ".join([str(x) for x in self.all_masked_lm_positions]))
        s += "masked_lm_labels: %s\n" % (
            " ".join([printable_text(x) for x in self.all_masked_lm_labels]))
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


def create_int_feature(values):
    feature = tf.train.Feature(
        int64_list=tf.train.Int64List(value=list(values)))
    return feature


def create_float_feature(values):
    feature = tf.train.Feature(
        float_list=tf.train.FloatList(value=list(values)))
    return feature


def gen_samples(sequences,
                dupe_factor,
                masked_lm_prob,
                max_predictions_per_seq,
                pool_size,
                rng,
                force_head=False):
    instances = []
    # create train
    for step in range(dupe_factor):
        start = time.time()
        for tokens in sequences:
            in_tokens = tokens[0]
            (address, in_tokens, in_masked_lm_positions,
             in_masked_lm_labels) = create_masked_lm_predictions(
                in_tokens, masked_lm_prob, max_predictions_per_seq, rng)

            out_tokens = tokens[1]
            (address, out_tokens, out_masked_lm_positions,
             out_masked_lm_labels) = create_masked_lm_predictions(
                out_tokens, masked_lm_prob, max_predictions_per_seq, rng)

            all_tokens = tokens[2]
            (address, all_tokens, all_masked_lm_positions,
             all_masked_lm_labels) = create_masked_lm_predictions(
                all_tokens, masked_lm_prob, max_predictions_per_seq, rng)

            instance = TrainingInstance(
                address=address,
                in_tokens=in_tokens,
                in_masked_lm_positions=in_masked_lm_positions,
                in_masked_lm_labels=in_masked_lm_labels,
                out_tokens=out_tokens,
                out_masked_lm_positions=out_masked_lm_positions,
                out_masked_lm_labels=out_masked_lm_labels,
                all_tokens=all_tokens,
                all_masked_lm_positions=all_masked_lm_positions,
                all_masked_lm_labels=all_masked_lm_labels)

            instances.append(instance)

        end = time.time()
        cost = end - start
        print("step=%d, time=%.2f" % (step, cost))

    print("=======Finish========")

    return instances


def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, rng):
    """Creates the predictions for the masked LM objective."""

    address = tokens[0][0]
    cand_indexes = []
    for (i, token) in enumerate(tokens):
        cand_indexes.append(i)

    rng.shuffle(cand_indexes)
    output_tokens = [list(i) for i in tokens]  # note that change the value of output_tokens will also change tokens
    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(len(tokens) * masked_lm_prob)))
    masked_lms = []
    covered_indexes = set()
    for index in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if index in covered_indexes:
            continue
        covered_indexes.add(index)
        masked_token = "[MASK]"
        masked_lms.append(MaskedLmInstance(index=index, label=tokens[index][0]))
        output_tokens[index][0] = masked_token

    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)
    return (address, output_tokens, masked_lm_positions, masked_lm_labels)


def create_embedding_predictions(tokens):
    """Creates the predictions for the masked LM objective."""
    address = tokens[0][0]
    output_tokens = tokens
    masked_lm_positions = []
    masked_lm_labels = []
    return (address, output_tokens, masked_lm_positions, masked_lm_labels)


def gen_embedding_samples(sequences):
    instances = []
    # create train
    start = time.time()
    for tokens in sequences:

        in_tokens = tokens[0]
        (address, in_tokens, in_masked_lm_positions,
         in_masked_lm_labels) = create_embedding_predictions(in_tokens)

        out_tokens = tokens[1]
        (address, out_tokens, out_masked_lm_positions,
         out_masked_lm_labels) = create_embedding_predictions(out_tokens)

        all_tokens = tokens[2]
        (address, all_tokens, all_masked_lm_positions,
         all_masked_lm_labels) = create_embedding_predictions(all_tokens)

        instance = TrainingInstance(
            address=address,
            in_tokens=in_tokens,
            in_masked_lm_positions=in_masked_lm_positions,
            in_masked_lm_labels=in_masked_lm_labels,
            out_tokens=out_tokens,
            out_masked_lm_positions=out_masked_lm_positions,
            out_masked_lm_labels=out_masked_lm_labels,
            all_tokens=all_tokens,
            all_masked_lm_positions=all_masked_lm_positions,
            all_masked_lm_labels=all_masked_lm_labels
        )
        instances.append(instance)

    end = time.time()
    print("=======Finish========")
    print("cost time:%.2f" % (end - start))
    return instances


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



def write_instance_to_example_files(instances, max_seq_length,
                                    max_predictions_per_seq, vocab,
                                    output_files):
    """Create TF example files from `TrainingInstance`s."""
    writers = []
    for output_file in output_files:
        writers.append(tf.python_io.TFRecordWriter(output_file))

    writer_index = 0
    total_written = 0

    for inst_index in tqdm(range(len(instances))):
        instance = instances[inst_index]
        address = vocab.convert_tokens_to_ids(instance.address)

        # ---------------- in sequence -------------------
        in_token_ids = vocab.convert_tokens_to_ids(instance.in_tokens)
        in_counts = instance.in_cnts
        in_io_flags = instance.in_io_flags
        in_block_timestamps = instance.in_block_timestamps
        in_values = instance.in_values
        in_positions = convert_timestamp_to_position(in_block_timestamps)

        in_mask = [1] * len(in_token_ids)
        assert len(in_token_ids) <= max_seq_length
        assert len(in_io_flags) <= max_seq_length
        assert len(in_counts) <= max_seq_length
        assert len(in_values) <= max_seq_length
        assert len(in_positions) <= max_seq_length

        in_token_ids += [0] * (max_seq_length - len(in_token_ids))
        in_io_flags += [0] * (max_seq_length - len(in_io_flags))
        in_counts += [0] * (max_seq_length - len(in_counts))
        in_values += [0] * (max_seq_length - len(in_values))
        in_positions += [0] * (max_seq_length - len(in_positions))
        in_mask += [0] * (max_seq_length - len(in_mask))

        assert len(in_token_ids) == max_seq_length
        assert len(in_io_flags) == max_seq_length
        assert len(in_counts) == max_seq_length
        assert len(in_values) == max_seq_length
        assert len(in_positions) == max_seq_length
        assert len(in_mask) == max_seq_length

        in_masked_lm_positions = list(instance.in_masked_lm_positions)
        in_masked_lm_ids = vocab.convert_tokens_to_ids(instance.in_masked_lm_labels)
        in_masked_lm_weights = [1.0] * len(in_masked_lm_ids)

        in_masked_lm_positions += [0] * (max_predictions_per_seq - len(in_masked_lm_positions))
        in_masked_lm_ids += [0] * (max_predictions_per_seq - len(in_masked_lm_ids))
        in_masked_lm_weights += [0.0] * (max_predictions_per_seq - len(in_masked_lm_weights))

        # ---------------- out sequence ------------------
        out_token_ids = vocab.convert_tokens_to_ids(instance.out_tokens)
        out_counts = instance.out_cnts
        out_io_flags = instance.out_io_flags
        out_block_timestamps = instance.out_block_timestamps
        out_values = instance.out_values
        out_positions = convert_timestamp_to_position(out_block_timestamps)

        out_mask = [1] * len(out_token_ids)
        assert len(out_token_ids) <= max_seq_length
        assert len(out_io_flags) <= max_seq_length
        assert len(out_counts) <= max_seq_length
        assert len(out_values) <= max_seq_length
        assert len(out_positions) <= max_seq_length

        out_token_ids += [0] * (max_seq_length - len(out_token_ids))
        out_io_flags += [0] * (max_seq_length - len(out_io_flags))
        out_counts += [0] * (max_seq_length - len(out_counts))
        out_values += [0] * (max_seq_length - len(out_values))
        out_positions += [0] * (max_seq_length - len(out_positions))
        out_mask += [0] * (max_seq_length - len(out_mask))

        assert len(out_token_ids) == max_seq_length
        assert len(out_io_flags) == max_seq_length
        assert len(out_counts) == max_seq_length
        assert len(out_values) == max_seq_length
        assert len(out_positions) == max_seq_length
        assert len(out_mask) == max_seq_length

        out_masked_lm_positions = list(instance.out_masked_lm_positions)
        out_masked_lm_ids = vocab.convert_tokens_to_ids(instance.out_masked_lm_labels)
        out_masked_lm_weights = [1.0] * len(out_masked_lm_ids)

        out_masked_lm_positions += [0] * (max_predictions_per_seq - len(out_masked_lm_positions))
        out_masked_lm_ids += [0] * (max_predictions_per_seq - len(out_masked_lm_ids))
        out_masked_lm_weights += [0.0] * (max_predictions_per_seq - len(out_masked_lm_weights))

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

        all_masked_lm_positions = list(instance.all_masked_lm_positions)
        all_masked_lm_ids = vocab.convert_tokens_to_ids(instance.all_masked_lm_labels)
        all_masked_lm_weights = [1.0] * len(all_masked_lm_ids)

        all_masked_lm_positions += [0] * (max_predictions_per_seq - len(all_masked_lm_positions))
        all_masked_lm_ids += [0] * (max_predictions_per_seq - len(all_masked_lm_ids))
        all_masked_lm_weights += [0.0] * (max_predictions_per_seq - len(all_masked_lm_weights))

        # feature generation

        features = collections.OrderedDict()
        features["address"] = create_int_feature(address)
        # in sequence
        features["in_token_ids"] = create_int_feature(in_token_ids)
        features["in_positions"] = create_int_feature(in_positions)
        features["in_io_flags"] = create_int_feature(in_io_flags)
        features["in_counts"] = create_int_feature(in_counts)
        features["in_values"] = create_int_feature(in_values)
        features["in_mask"] = create_int_feature(in_mask)
        features["in_masked_lm_positions"] = create_int_feature(in_masked_lm_positions)
        features["in_masked_lm_ids"] = create_int_feature(in_masked_lm_ids)
        features["in_masked_lm_weights"] = create_float_feature(in_masked_lm_weights)

        # out sequence
        features["out_token_ids"] = create_int_feature(out_token_ids)
        features["out_positions"] = create_int_feature(out_positions)
        features["out_io_flags"] = create_int_feature(out_io_flags)
        features["out_counts"] = create_int_feature(out_counts)
        features["out_values"] = create_int_feature(out_values)
        features["out_mask"] = create_int_feature(out_mask)
        features["out_masked_lm_positions"] = create_int_feature(out_masked_lm_positions)
        features["out_masked_lm_ids"] = create_int_feature(out_masked_lm_ids)
        features["out_masked_lm_weights"] = create_float_feature(out_masked_lm_weights)

        # all sequence
        features["all_token_ids"] = create_int_feature(all_token_ids)
        features["all_positions"] = create_int_feature(all_positions)
        features["all_io_flags"] = create_int_feature(all_io_flags)
        features["all_counts"] = create_int_feature(all_counts)
        features["all_values"] = create_int_feature(all_values)
        features["all_mask"] = create_int_feature(all_mask)
        features["all_masked_lm_positions"] = create_int_feature(all_masked_lm_positions)
        features["all_masked_lm_ids"] = create_int_feature(all_masked_lm_ids)
        features["all_masked_lm_weights"] = create_float_feature(all_masked_lm_weights)

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



def main():

    vocab = FreqVocab()
    print("===========Load Sequence===========")
    with open("./data/eoa2seq_" + FLAGS.source_bizdate + ".pkl", "rb") as f:
        eoa2seq = pkl.load(f)

    print("number of target user account:", len(eoa2seq))
    vocab.update(eoa2seq)
    # generate mapping
    vocab.generate_vocab()

    # save vocab
    print("token_size:{}".format(len(vocab.vocab_words)))
    vocab_file_name = FLAGS.data_dir + FLAGS.vocab_filename + "." + FLAGS.bizdate
    print('vocab pickle file: ' + vocab_file_name)
    with open(vocab_file_name, 'wb') as output_file:
        pkl.dump(vocab, output_file, protocol=2)

    print("===========Original===========")
    length_list = []
    for eoa in eoa2seq.keys():
        seq = eoa2seq[eoa]
        length_list.append(len(seq))

    length_list = np.array(length_list)
    print("Median:", np.median(length_list))
    print("Mean:", np.mean(length_list))
    print("Seq num:", len(length_list))


    # clip
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


    if FLAGS.do_embed:
        print("===========Generate Embedding Samples==========")
        write_instance = gen_embedding_samples(seqs)
        output_filename = FLAGS.data_dir + "embed.tfrecord" + "." + FLAGS.bizdate
        tf.logging.info("*** Writing to output embedding files ***")
        tf.logging.info("  %s", output_filename)

        write_instance_to_example_files(write_instance, FLAGS.max_seq_length,
                                        MAX_PREDICTIONS_PER_SEQ, vocab,
                                        [output_filename])

    seqs = np.random.permutation(seqs)

    if FLAGS.do_eval:  # select 20% for testing
        print("========Generate Evaluation Samples========")
        eval_seqs = seqs[:round(len(seqs) * 0.2)]
        seqs = seqs[round(len(seqs) * 0.2):]

        eval_normal_instances = gen_samples(eval_seqs,
                                            dupe_factor=FLAGS.dupe_factor,
                                            masked_lm_prob=FLAGS.masked_lm_prob,
                                            max_predictions_per_seq=MAX_PREDICTIONS_PER_SEQ,
                                            pool_size=FLAGS.pool_size,
                                            rng=rng,
                                            force_head=False)

        eval_write_instance = eval_normal_instances
        rng.shuffle(eval_write_instance)
        eval_output_filename = FLAGS.data_dir + "test.tfrecord" + "." + FLAGS.bizdate
        tf.logging.info("*** Writing to Testing files ***")
        tf.logging.info("  %s", eval_output_filename)

        write_instance_to_example_files(eval_write_instance, FLAGS.max_seq_length,
                                        MAX_PREDICTIONS_PER_SEQ, vocab,
                                        [eval_output_filename])

    print("========Generate Training Samples========")
    normal_instances = gen_samples(seqs,
                                   dupe_factor=FLAGS.dupe_factor,
                                   masked_lm_prob=FLAGS.masked_lm_prob,
                                   max_predictions_per_seq=MAX_PREDICTIONS_PER_SEQ,
                                   pool_size=FLAGS.pool_size,
                                   rng=rng,
                                   force_head=False)

    write_instance = normal_instances
    rng.shuffle(write_instance)

    output_filename = FLAGS.data_dir + "train.tfrecord" + "." + FLAGS.bizdate
    tf.logging.info("*** Writing to Training files ***")
    tf.logging.info("  %s", output_filename)

    write_instance_to_example_files(write_instance, FLAGS.max_seq_length,
                                    MAX_PREDICTIONS_PER_SEQ, vocab,
                                    [output_filename])

    return


if __name__ == '__main__':
    main()


