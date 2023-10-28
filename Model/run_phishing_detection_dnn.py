import pickle as pkl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score, roc_curve, auc, precision_recall_curve
import os
import time

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_bool("visual", False, "whether to do visualization or not")
flags.DEFINE_string("init_checkpoint", None, "Initial checkpoint (usually from a pre-trained BERT model).")
flags.DEFINE_integer("batch_size", 256, "")
flags.DEFINE_integer("epoch", 2, "")


class DNN(object):

    def __init__(self):
        # configuration
        self.model_config = {
            "hidden": 256,
            "learning_rate": 5e-4
        }

    def fit(self, features):

        label = features[:, 0]
        embedding = features[:, 1:]

        self.label = tf.squeeze(label)
        self.build_fcn_net(embedding)
        self.loss_op()

    def predict(self, features):
        label = features[:, 0]
        embedding = features[:, 1:]
        y_hat = self.build_fcn_net(embedding)

        return y_hat, label

    def build_fcn_net(self, embedding):
        with tf.name_scope("Fully_connected_layer"):
            inp = embedding
            dnn1 = tf.layers.dense(inp, self.model_config["hidden"], activation=tf.nn.relu, name='f1', reuse=tf.AUTO_REUSE)
            dnn2 = tf.layers.dense(dnn1, self.model_config["hidden"], activation=tf.nn.relu, name='f2', reuse=tf.AUTO_REUSE)
            self.logit = tf.squeeze(tf.layers.dense(dnn1 + dnn2, 1, activation=None, name='logit', reuse=tf.AUTO_REUSE))

        self.y_hat = tf.sigmoid(self.logit)
        return self.y_hat

    def loss_op(self):
        with tf.name_scope('Metrics'):
            print(self.logit)
            print(self.label)
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.logit))
            tf.summary.scalar('loss', self.loss)
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.model_config["learning_rate"]).minimize(self.loss)

        self.merged = tf.summary.merge_all()
        return


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

def load_embedding():

    # must have checkpoint
    if FLAGS.init_checkpoint == None:
        raise ValueError("Must need a checkpoint for evaluation")

    checkpoint_name = FLAGS.init_checkpoint.split("/")[0]
    model_index = str(FLAGS.init_checkpoint.split("/")[-1].split("_")[1])
    embeddings = np.load("./inter_data/embedding_" + checkpoint_name + "_" + model_index + ".npy")
    address_for_embedding = np.load("./inter_data/address_" + checkpoint_name + "_" + model_index + ".npy")

    # group by embedding according to address
    address_to_embedding = {}

    for i in range(len(address_for_embedding)):
        address = address_for_embedding[i]
        embedding = embeddings[i]
        try:
            address_to_embedding[address].append(embedding)
        except:
            address_to_embedding[address] = [embedding]

    # group to one
    address_list = []
    embedding_list = []

    for addr, embeds in address_to_embedding.items():
        address_list.append(addr)
        if len(embeds) > 1:
            embedding_list.append(np.mean(embeds, axis=0))
        else:
            embedding_list.append(embeds[0])

    # final embedding table
    X = np.array(np.squeeze(embedding_list))

    return X, address_list

if __name__ == '__main__':

    phisher_account = pd.read_csv("../Data/phisher_account.txt", names=["account"])
    phisher_account_set = set(phisher_account.account.values)

    exp_account = pd

    X, address_list = load_embedding()

    y = []
    for addr in address_list:
        if addr in phisher_account_set:
            y.append(1)
        else:
            y.append(0)

    y = np.expand_dims(y, axis=1)
    X = np.array(X)

    print(X.shape)
    print(y.shape)
    print(np.sum(y))
    # raise ValueError("Finish")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    train_set = np.concatenate([y_train, X_train], axis=1)
    test_set = np.concatenate([y_test, X_test], axis=1)

    # build dataset
    train_dataset = tf.data.Dataset.from_tensor_slices(train_set).repeat(FLAGS.epoch).shuffle(100).batch(batch_size=FLAGS.batch_size)
    train_iterator = train_dataset.make_one_shot_iterator()
    train_features = train_iterator.get_next()

    # testing dataset
    test_dataset = tf.data.Dataset.from_tensor_slices(test_set).batch(batch_size=FLAGS.batch_size)
    test_iterator = test_dataset.make_one_shot_iterator()
    test_features = test_iterator.get_next()

    print(train_features)
    # build DNN model
    model = DNN()
    model.fit(train_features)

    # saver = tf.train.Saver(max_to_keep=30)
    # save_iter = int(sample_num / FLAGS.batch_size)
    pred_probas = []
    labels = []

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    losses = []
    iter = 0
    start = time.time()

    while True:
        # for i in range(100):
        try:
            _, loss = sess.run([model.train_op, model.loss])
            losses.append(loss)

            if iter % 100 == 0:
                end = time.time()
                loss = np.mean(losses)
                print("iter=%d, loss=%f, time=%.2fs" % (iter, loss, end - start))
                losses = []
                start = time.time()

            iter += 1

            # if iter % save_iter == 0 and iter > 0:
            #     saver.save(sess, os.path.join(FLAGS.checkpointDir, "model_" + str(round(iter / save_iter))))
            #     pass

        except Exception as e:
            # print(e)
            break

    # predict
    y_hat, label = model.predict(test_features)

    y_hat_list = []
    label_list = []
    while True:
        try:
            y_hat_v, label_v = sess.run([y_hat, label])
            y_hat_list += list(y_hat_v)
            label_list += list(label_v)
        except Exception as e:
            break

    # y_hat_list = np.array(y_hat_list).reshape([-1])
    # label_list = np.array(label_list).reshape([-1])
    print(len(y_hat_list))
    print(np.sum(y_hat_list))
    print(np.sum(label_list))

    # print("================ROC Curve====================")
    model_index = str(FLAGS.init_checkpoint.split("/")[-1].split("_")[1])
    print("model_index:", model_index)
    fpr, tpr, thresholds = roc_curve(label_list, y_hat_list, pos_label=1)
    print("AUC=", auc(fpr, tpr))

    for threshold in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        print("threshold =", threshold)
        y_pred = np.zeros_like(y_hat_list)
        y_pred[np.where(np.array(y_hat_list) >= threshold)[0]] = 1
        print(np.sum(y_pred))
        print(classification_report(label_list, y_pred, digits=4))

