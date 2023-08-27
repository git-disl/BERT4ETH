import pandas as pd
import numpy as np
import pickle as pkl
from tqdm import tqdm
from numpy import dot
from numpy.linalg import norm
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("metric", "euclidean", "")
flags.DEFINE_string("ens_dataset", "../Data/dean_all_ens_pairs.csv", "")
flags.DEFINE_string("algo", "", "algorithm for embedding generation" )
flags.DEFINE_string("model_index", None, "model index")
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
    indices = list(sorted_df["idx"])
    indices.remove(idx)  # exclude self distance

    if len(include_idx_mask) > 0:
        # filter indices
        indices_tmp = []
        for i, res_idx in enumerate(indices):
            if res_idx in include_idx_mask:
                indices_tmp.append(res_idx)
        indices = indices_tmp
    return indices

def get_rank(X, query_idx, target_idx, include_idx_mask=[]):
    indices = get_neighbors(X, query_idx, FLAGS.metric, include_idx_mask)
    if len(indices) > 0 and target_idx in indices:
        trg_idx = indices.index(target_idx)
        return trg_idx+1, len(indices)
    else:
        return None, len(indices)


def get_deposit_indices(addr2idx, tq, tup, f_id):
    """Extract the possible set of deposit address candidates for each heuristic record given different temporal filtering options"""
    if f_id == "day":
        d_addr_set = tq.get_possible_deposits(tup, 86400)
        anonymity_set_size = len(d_addr_set)
    elif f_id == "week":
        d_addr_set = tq.get_possible_deposits(tup, 7 * 86400)
        anonymity_set_size = len(d_addr_set)
    elif f_id == "past":
        d_addr_set = tq.get_possible_deposits(tup, None)
        anonymity_set_size = len(d_addr_set)
    else:
        # d_addr_set = []
        # anonymity_set_size = len(a2v_obj.addr_to_embedd) - 1
        raise ValueError("Please choose write f_id")

    d_addr_idx = []
    for addr in d_addr_set:
        if addr in addr2idx:
            d_addr_idx.append(addr2idx[addr])
    return d_addr_idx, anonymity_set_size


def clean_heuristics(tornado_pairs, verbose):
    """Removing loops and Tornado contract addresses from the set of heuristics"""
    orig_size = len(tornado_pairs)
    # removing loops
    tornado_pairs = tornado_pairs[tornado_pairs["sender"] != tornado_pairs["receiver"]]
    if verbose:
        print("Loops removed:", len(tornado_pairs) / orig_size)
    # removing Tornado cash 0.1ETH address
    tornado_pairs = tornado_pairs[tornado_pairs["receiver"] != "0x12d66f87a04a9e220743712ce6d9bb1b5616b8fc"]
    if verbose:
        print("Tornado address removed:", len(tornado_pairs) / orig_size)
    return tornado_pairs


def temporal_filter(df, max_time):
    if max_time != None:
        return df[df["timeStamp"] <= max_time]
    else:
        return df


class TornadoQueries():
    def __init__(self, mixer_str_value="0.1", data_type="all", data_folder="../Data/tornado_raw_data", max_time=None, verbose=True):
        self.mixer_str_value = mixer_str_value
        self.data_type = data_type
        self.data_folder = data_folder
        self.max_time = max_time
        self.verbose = verbose
        self.history_df = temporal_filter(self._load_history(), self.max_time)
        self.tornado_pairs = temporal_filter(self._load_heuristics(), self.max_time)
        self.tornado_tuples = list(
            zip(self.tornado_pairs["sender"], self.tornado_pairs["receiver"], self.tornado_pairs["withdHash"],
                self.tornado_pairs["timeStamp"]))
        if self.verbose:
            print("history", self.history_df.shape)
            print("pairs", self.tornado_pairs.shape)

    def _load_history(self):
        history_df = pd.read_csv("%s/tornadoFullHistoryMixer_%sETH.csv" % (self.data_folder, self.mixer_str_value))
        history_df = history_df.sort_values("timeStamp")
        history_df["contrib"] = history_df["action"].apply(lambda x: 1 if x == "d" else 0)
        history_df["num_deps"] = history_df["contrib"].cumsum()
        if "index" in history_df.columns:
            history_df = history_df.drop("index", axis=1)
        self.tornado_hash_time = dict(zip(history_df["txHash"], history_df["timeStamp"]))
        return history_df

    def _load_heuristics(self):
        if self.data_type == "heur2":
            tornado_pairs = pd.read_csv("%s/heuristic2Mixer_%sETH.csv" % (self.data_folder, self.mixer_str_value))
        elif self.data_type == "heur3":
            tornado_pairs = pd.read_csv("%s/heuristic3Mixer_%sETH.csv" % (self.data_folder, self.mixer_str_value))
        else:
            heur2 = pd.read_csv("%s/heuristic2Mixer_%sETH.csv" % (self.data_folder, self.mixer_str_value))
            heur3 = pd.read_csv("%s/heuristic3Mixer_%sETH.csv" % (self.data_folder, self.mixer_str_value))
            tornado_pairs = pd.concat([heur2, heur3])
        tornado_pairs = tornado_pairs.drop_duplicates()
        tornado_pairs["timeStamp"] = tornado_pairs["withdHash"].apply(lambda x: self.tornado_hash_time[x])
        tornado_pairs = clean_heuristics(tornado_pairs, self.verbose)
        if "Unnamed: 0" in tornado_pairs.columns:
            tornado_pairs = tornado_pairs.drop("Unnamed: 0", axis=1)
        return tornado_pairs

    def plot_num_deposits(self, show_heuristics=True, linew=3, msize=10):
        """Visualize the temporal distribution of the found withdraw-deposit address pairs (show_heuristics=True) or the number of active deposits (show_heuristics=False)"""
        df = self.history_df
        if show_heuristics:
            plt.plot(pd.to_datetime(self.tornado_pairs["timeStamp"], unit='s'),
                     np.ones(len(self.tornado_pairs)) * float(self.mixer_str_value), 'x',
                     label="%sETH" % self.mixer_str_value, linewidth=linew, markersize=msize)
        else:
            plt.plot(pd.to_datetime(df["timeStamp"], unit='s'), df["num_deps"], label="%sETH" % self.mixer_str_value,
                     linewidth=linew, markersize=msize)
        plt.xticks(rotation=90)

    def get_possible_deposits(self, tornado_tuple, time_interval=None):
        """Get possible deposit address set for a withdraw transaction. Provide the 'time_interval' in seconds if you have some temporal assumption on the timestamp of the deposit."""
        d, w, h, time_bound = tornado_tuple
        filtered_df = self.history_df[(self.history_df["action"] == "d") & (self.history_df["timeStamp"] <= time_bound)]
        if time_interval != None:
            filtered_df = filtered_df[filtered_df["timeStamp"] >= (time_bound - time_interval)]
        return list(filtered_df["account"].unique())



def load_embedding():

    if FLAGS.algo == "BERT4ETH":
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

    # load embedding and map address to int
    X, address_list = load_embedding()

    # all_address_list = set(address_list)
    # tornado_address_set = set(pd.read_csv("../Data/tornado_dataset/raw_data/tornado_account.csv").address.values)

    cnt = 0
    address_to_idx = {}
    idx_to_address = {}
    for address in address_list:
        address_to_idx[address] = cnt
        idx_to_address[cnt] = address
        cnt += 1

    # generate pairs
    MAX_TIME = 1587000000 # 2020-04-15 21:20:00
    tq0_1 = TornadoQueries(mixer_str_value="0.1", max_time=MAX_TIME)
    tq1 = TornadoQueries(mixer_str_value="1", max_time=MAX_TIME)
    tq10 = TornadoQueries(mixer_str_value="10", max_time=MAX_TIME)
    queries = [tq0_1, tq1, tq10]

    # # Evaluate embeddings for Tornado withdraw-deposit address pairs
    pairs = []
    for tq in queries:
        pairs.append(tq.tornado_pairs[["sender", "receiver"]])
    pairs = pd.concat(pairs).reset_index(drop=True)
    pairs = pairs.drop_duplicates()
    print("Evaluated withdraw-deposit:", pairs.shape)

    def run_tornado(query_objects, filters=["none", "past", "week", "day"]):
        """Evaluate representations for Tornado withdraw-deposit heuristics"""
        res = []
        pbar = tqdm(total=len(query_objects))
        for tq in query_objects:
            records = []
            for tup in tq.tornado_tuples:
                d_addr, w_addr = tup[0], tup[1]
                if d_addr in address_to_idx and w_addr in address_to_idx:
                    d_idx, w_idx = address_to_idx[d_addr], address_to_idx[w_addr]
                    for f_id in filters:
                        # anonymity set size is needed without filters
                        d_set_idx, size_1 = get_deposit_indices(address_to_idx, tq, tup, f_id)
                        rank, size_2 = get_rank(X, w_idx, d_idx, d_set_idx)
                        num_set = max(size_1, size_2)
                        records.append((tup[3], w_idx, d_idx, rank, num_set, f_id))
            df = pd.DataFrame(records, columns=["timestamp","query_idx","target_idx","rank","set_size","filter"])
            df["mixer"] = tq.mixer_str_value
            res.append(df)
            pbar.update(1)
        pbar.close()
        df = pd.concat(res)
        df["query_addr"] = df["query_idx"].apply(lambda x: idx_to_address[x])
        df["target_addr"] = df["target_idx"].apply(lambda x: idx_to_address[x])
        return df.drop(["query_idx","target_idx"], axis=1)

    def get_avg_rank(df, keys=["filter"]):
        """Aggregate results for independent experiments"""
        target_cols = ["rank", "set_size"]
        if "mixer" in df.columns and "mixer" not in keys:
            keys.append("mixer")
        for col in ["rank_ratio", "auc"]:
            if col in df.columns:
                target_cols.append(col)
        # aggregate for independent experiments
        mean_result = df.groupby(["query_addr", "target_addr"] + keys)[target_cols].mean().reset_index()
        median_result = df.groupby(["query_addr", "target_addr"] + keys)[target_cols].median().reset_index()
        # aggregate for address pairs
        perf_mean = mean_result.groupby(keys)[target_cols].mean().reset_index()
        perf_median = median_result.groupby(keys)[target_cols].median().reset_index()
        return perf_mean, perf_median

    tornado_result = run_tornado(queries, filters=["past", "week", "day"])
    tornado_perf_mean, tornado_perf_median = get_avg_rank(tornado_result)
    print("================MEAN================")
    print(tornado_perf_mean)
    print("===============MEDIAN================")
    print(tornado_perf_median)

    # EXPORT

    output_file = "data/" + FLAGS.algo + "_dean_Tornado_" + FLAGS.model_index + ".csv"
    tornado_result.to_csv(output_file, index=False)


if __name__ == '__main__':
    main()


