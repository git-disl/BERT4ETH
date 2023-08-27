import pickle as pkl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score, roc_curve, auc, precision_recall_curve
import lightgbm as lgb

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_bool("visual", False, "whether to do visualization or not")
flags.DEFINE_string("algo", None, "algorithm for embedding generation" )
flags.DEFINE_string("model_index", None, "model index")


def load_embedding():

    if FLAGS.algo == "deepwalk":
        embeddings = np.load("deepwalk/data/deepwalk_20_10_20220925_embedding.npy")
        address_for_embedding = np.load("deepwalk/data/deepwalk_20_10_20220925_address.npy")

    elif FLAGS.algo == "trans2vec":
        embeddings = np.load("trans2vec/data/tran2vec_20_10_0.5_1_20221010_embedding.npy")
        address_for_embedding = np.load("trans2vec/data/tran2vec_20_10_0.5_1_20221010_address.npy")

    elif FLAGS.algo == "bert4eth":
        embeddings = np.load("BERT4ETH/data/bert_embedding_" + FLAGS.model_index + ".npy")
        address_for_embedding = np.load("BERT4ETH/data/address_for_embed_" + FLAGS.model_index + ".npy")

    elif FLAGS.algo == "diff2vec":
        embeddings = np.load("diff2vec/data/diff2vec_10_20_20220925_embedding.npy")
        address_for_embedding = np.load("diff2vec/data/diff2vec_10_20_20220925_address.npy")

    elif FLAGS.algo == "role2vec":
        embeddings = np.load("role2vec/data/role2vec_10_20_20220916_embedding.npy")
        address_for_embedding = np.load("role2vec/data/role2vec_10_20_20220916_address.npy")

    elif FLAGS.algo == "sage":
        embeddings = np.load("dgl_gnn/data/sage_6_20220915_embedding.npy")
        address_for_embedding = np.load("dgl_gnn/data/sage_6_20220915_address.npy")

    elif FLAGS.algo == "gat":
        embeddings = np.load("dgl_gnn/data/gat_7_20220915_embedding.npy")
        address_for_embedding = np.load("dgl_gnn/data/gat_7_20220915_address.npy")

    elif FLAGS.algo == "gcn":
        embeddings = np.load("dgl_gnn/data/gcn_5_20220915_embedding.npy")
        address_for_embedding = np.load("dgl_gnn/data/gcn_5_20220915_address.npy")

    else:
        raise ValueError("should choose right algo..")

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

def main():

    phisher_account = pd.read_csv("Data/phisher_account.txt", names=["account"])
    phisher_account_set = set(phisher_account.account.values)

    X, address_list = load_embedding()

    y = []
    for addr in address_list:
        if addr in phisher_account_set:
            y.append(1)
        else:
            y.append(0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier(n_estimators=50, criterion='entropy', random_state=0)
    model.fit(X_train, y_train)

    y_test_proba = model.predict_proba(X_test)[:, 1]
    # rf_output = np.array(list(zip(y_test_proba, y_test)))
    # if FLAGS.algo in ["sage", "gcn", "gat", "gin"]:
    #     np.save("dgl_gnn/data/" + FLAGS.algo + "_phisher_account_output_" + FLAGS.dataset + ".npy", rf_output)
    # else:
    #     np.save(FLAGS.algo + "/data/phisher_account_output_"+ FLAGS.dataset +".npy", rf_output)

    print("=============Precision-Recall Curve=============")
    precision, recall, thresholds_pr = precision_recall_curve(y_test, y_test_proba)
    plt.figure("P-R Curve")
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.plot(recall, precision)
    plt.show()

    # print("================ROC Curve====================")
    print("model_index:", FLAGS.model_index)
    fpr, tpr, thresholds = roc_curve(y_test, y_test_proba, pos_label=1)
    print("AUC=", auc(fpr, tpr))

    plt.figure("ROC Curve")
    plt.title("ROC Curve")
    plt.xlabel("fpr")
    plt.ylabel("tpr")
    plt.plot(fpr, tpr)
    plt.show()

    for threshold in [0.1, 0.15, 0.2]:
        print("threshold =", threshold)
        y_pred = np.zeros_like(y_test_proba)
        y_pred[np.where(np.array(y_test_proba) >= threshold)[0]] = 1
        print(np.sum(y_pred))
        print(classification_report(y_test, y_pred, digits=4))

    if FLAGS.visual:
        # visualization
        y = np.array(y)
        p_idx = np.where(y == 1)[0]
        n_idx = np.where(y == 0)[0]
        X_phisher = X[p_idx]
        X_normal = X[n_idx]

        permutation = np.random.permutation(len(X_normal))
        X_normal_sample = X_normal[permutation[:10000]]
        X4tsne = np.concatenate([X_normal_sample, X_phisher], axis=0)
        tsne = TSNE(n_components=2, init="random")
        X_tsne = tsne.fit_transform(X4tsne)

        # plot
        plt.figure(figsize=(8, 6), dpi=80)
        plt.scatter(x=X_tsne[:10000, 0], y=X_tsne[:10000, 1], marker=".")
        plt.scatter(x=X_tsne[10000:, 0], y=X_tsne[10000:, 1], marker=".", color="orange")
        plt.show()

        plt.figure(figsize=(8, 6), dpi=80)
        plt.scatter(x=X_tsne[:10000, 0], y=X_tsne[:10000, 1], marker=".")
        plt.show()


if __name__ == '__main__':
    main()
