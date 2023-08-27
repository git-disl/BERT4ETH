import pandas as pd
import numpy as np
import pickle as pkl
from tqdm import tqdm
from numpy import dot
from numpy.linalg import norm
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("algo", "", "algorithm for embedding generation" )
flags.DEFINE_string("model_index", None, "model index")
flags.DEFINE_string("metric", "euclidean", "")
flags.DEFINE_string("ens_dataset", "../Data/dean_all_ens_pairs.csv", "")
flags.DEFINE_integer("max_cnt", 2, "")

def euclidean_dist(a, b):
    return np.sqrt(np.sum(np.square(a-b)))

def cosine_dist(a, b):
    return 1-dot(a, b)/(norm(a)*norm(b)) # notice, here need 1-

def cosine_dist_multi(a, b):
    num = dot(a, b.T)
    denom = norm(a) * norm(b, axis=1)
    res = num/denom
    return -1 * res

def euclidean_dist_multi(a, b):
    return np.sqrt(np.sum(np.square(b-a), axis=1))

def get_neighbors(X, idx, metric="cosine" ,include_idx_mask=[]):
    a = X[idx, :]
    indices = list(range(X.shape[0]))
    if metric == "cosine":
        # dist = np.array([cosine_dist(a, X[i, :]) for i in indices])
        dist = cosine_dist_multi(a, X)
    elif metric == "euclidean":
        dist = euclidean_dist_multi(a, X)
    else:
        raise ValueError("Distance Metric Error")
    sorted_df = pd.DataFrame(list(zip(indices, dist)), columns=["idx", "dist"]).sort_values("dist")
    sorted_df = sorted_df.drop(index=idx) # exclude self distance
    indices = list(sorted_df["idx"])
    distances = list(sorted_df["dist"])

    if len(include_idx_mask) > 0:
        # filter indices
        indices_tmp = []
        distances_tmp = []
        for i, res_idx in enumerate(indices):
            if res_idx in include_idx_mask:
                indices_tmp.append(res_idx)
                distances_tmp.append(distances[i])
        indices = indices_tmp
        distances = distances_tmp
    return indices, distances


def get_rank(X, query_idx, target_idx, metric, include_idx_mask=[]):
    indices, distances = get_neighbors(X, query_idx, metric, include_idx_mask)
    if len(indices) > 0 and target_idx in indices:
        trg_idx = indices.index(target_idx)
        return trg_idx+1, distances[trg_idx], len(indices)
    else:
        return None, None, len(indices)


def generate_pairs(ens_pairs, min_cnt=2, max_cnt=2, mirror=True):
    """
    Generate testing pairs based on ENS name
    :param ens_pairs:
    :param min_cnt:
    :param max_cnt:
    :param mirror:
    :return:
    """
    pairs = ens_pairs.copy()
    ens_counts = pairs["name"].value_counts()
    address_pairs = []
    all_ens_names = []
    ename2addresses = {}
    for idx, row in pairs.iterrows():
        try:
            ename2addresses[row["name"]].append(row["address"]) # note: cannot use row.name
        except:
            ename2addresses[row["name"]] = [row["address"]]
    for cnt in range(min_cnt, max_cnt + 1):
        ens_names = list(ens_counts[ens_counts == cnt].index)
        all_ens_names += ens_names
        # convert to indices
        for ename in ens_names:
            addrs = ename2addresses[ename]
            for i in range(len(addrs)):
                for j in range(i + 1, len(addrs)):
                    addr1, addr2 = addrs[i], addrs[j]
                    address_pairs.append([addr1, addr2])
                    if mirror:
                        address_pairs.append([addr2, addr1])
    return address_pairs, all_ens_names


def load_embedding():


    if FLAGS.algo == "deepwalk":
        embeddings = np.load(FLAGS.embed_dir + "deepwalk/data/deepwalk_20_10_20220915_embedding.npy")
        address_for_embedding = np.load(FLAGS.embed_dir + "deepwalk/data/deepwalk_20_10_20220915_address.npy")

    elif FLAGS.algo == "trans2vec":
        embeddings = np.load(FLAGS.embed_dir + "trans2vec/data/tran2vec_20_10_0.5_1_20220915_embedding.npy")
        address_for_embedding = np.load(FLAGS.embed_dir + "trans2vec/data/tran2vec_20_10_0.5_1_20220915_address.npy")

    elif FLAGS.algo == "diff2vec":
        embeddings = np.load(FLAGS.embed_dir + "diff2vec/data/diff2vec_10_20_20220915_embedding.npy")
        address_for_embedding = np.load(FLAGS.embed_dir + "diff2vec/data/diff2vec_10_20_20220915_address.npy")

    elif FLAGS.algo == "role2vec":
        embeddings = np.load(FLAGS.embed_dir + "role2vec/data/role2vec_10_20_20220916_embedding.npy")
        address_for_embedding = np.load(FLAGS.embed_dir + "role2vec/data/role2vec_10_20_20220916_address.npy")

    elif FLAGS.algo == "sage":
        embeddings = np.load(FLAGS.graph_embed_dir + "/sage_4_link_predict_D1_2layer_embedding.npy")
        address_for_embedding = np.load(FLAGS.graph_embed_dir + "/sage_4_link_predict_D1_2layer_address.npy")

    elif FLAGS.algo == "gat":
        embeddings = np.load(FLAGS.graph_embed_dir + "/gat_4_link_predict_D1_2layer_embedding.npy")
        address_for_embedding = np.load(FLAGS.graph_embed_dir + "/gat_4_link_predict_D1_2layer_address.npy")

    elif FLAGS.algo == "gcn":
        embeddings = np.load(FLAGS.graph_embed_dir + "/gcn_4_link_predict_D1_2layer_embedding.npy")
        address_for_embedding = np.load(FLAGS.graph_embed_dir + "/gcn_4_link_predict_D1_2layer_address.npy")

    elif FLAGS.algo == "BERT":
        embeddings = np.load("BERT/data/bert_embedding_" + FLAGS.model_index + ".npy")
        address_for_embedding = np.load("BERT/data/address_for_embed_" + FLAGS.model_index + ".npy")

    elif FLAGS.algo == "BERT4ETH":
        embeddings = np.load("BERT4ETH/data/embedding_" + FLAGS.model_index + ".npy")
        address_for_embedding = np.load("BERT4ETH/data/address_" + FLAGS.model_index + ".npy")

    elif FLAGS.algo == "BERT4ETH_IOS":
        embeddings = np.load("BERT4ETH_IOS/data/embedding_" + FLAGS.model_index + ".npy")
        address_for_embedding = np.load("BERT4ETH_IOS/data/address_" + FLAGS.model_index + ".npy")

    elif FLAGS.algo == "BERT4ETH_ERC":
        embeddings = np.load("BERT4ETH_ERC/data/embedding_" + FLAGS.model_index + ".npy")
        address_for_embedding = np.load("BERT4ETH_ERC/data/address_" + FLAGS.model_index + ".npy")

    else:
        raise ValueError("should choose right algo..")

    # group by embedding according to address
    address_to_embedding = {}
    for i in range(len(address_for_embedding)):
        address = address_for_embedding[i]
        embedding = embeddings[i]
        # if address not in exp_addr_set:
        #     continue
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

    # load dataset
    ens_pairs = pd.read_csv(FLAGS.ens_dataset)
    max_ens_per_address = 1
    num_ens_for_addr = ens_pairs.groupby("address")["name"].nunique().sort_values(ascending=False).reset_index()
    excluded = list(num_ens_for_addr[num_ens_for_addr["name"] > max_ens_per_address]["address"])
    ens_pairs = ens_pairs[~ens_pairs["address"].isin(excluded)]
    address_pairs, all_ens_names = generate_pairs(ens_pairs, max_cnt=FLAGS.max_cnt)

    X, address_list = load_embedding()

    # map address to int
    cnt = 0
    address_to_idx = {}
    idx_to_address = {}
    for address in address_list:
        address_to_idx[address] = cnt
        idx_to_address[cnt] = address
        cnt += 1

    idx_pairs = []
    failed_address = []
    for pair in address_pairs:
        try:
            idx_pairs.append([address_to_idx[pair[0]], address_to_idx[pair[1]]])
        except:
            failed_address.append(pair[0])
            failed_address.append(pair[1])
            continue

    # calculate euclidean distance for ground-truth pairs
    ground_truth_euclidean_distance = []
    for pair in idx_pairs:
        src_id = pair[0]
        dst_id = pair[1]
        src_embedding = X[src_id]
        dst_embedding = X[dst_id]

        ground_truth_euclidean_distance.append(euclidean_dist(src_embedding, dst_embedding))

    print("pause")

    pbar = tqdm(total=len(idx_pairs))
    records = []
    for pair in idx_pairs:
        rank, dist, num_set = get_rank(X, pair[1], pair[0], FLAGS.metric)
        records.append((pair[1], pair[0], rank, dist, num_set, "none"))
        print(rank)
        pbar.update(1)

    result = pd.DataFrame(records, columns=["query_idx", "target_idx", "rank", "dist", "set_size", "filter"])
    result["query_addr"] = result["query_idx"].apply(lambda x: idx_to_address[x])
    result["target_addr"] = result["target_idx"].apply(lambda x: idx_to_address[x])
    result.drop(["query_idx", "target_idx"], axis=1)

    output_file = "data/" + FLAGS.algo + "_dean_ENS_" + FLAGS.model_index + ".csv"

    result.to_csv(output_file, index=False)

if __name__ == '__main__':
    main()


