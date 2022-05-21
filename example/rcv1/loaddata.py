import numpy as np
import pandas as pd


def load_data_rcv1(is_write=False):
    """load data from the files.
    Returns:
        train: the training data, a dict whose key is a did
            and the corresponding value is a list of its topics.
        test: the test data, a dict whose key is a did
            and the corresponding value is a list of its topics.
        tid_used: a list of tids appearing in at least 100 documents in the training data.
        topics: a dict whose key is a topic and the corresponding value
            is a list of dids in this topic.
        topics_used: a list of topics containing at least 200 documents.
        stem_tid: a dict map a did to corresponding stemmed term
    """
    def get_tids(line):
        """get did-tid tuples from a string"""
        did, tids = line.split('  ', 1)
        tids = list(map(lambda x: x[:x.index(':')], tids.split(' ')))
        return did, tids

    def select_tid(data):
        """select the tids that appear in at least 100 documents in the training data"""
        count = [0] * 47237
        for did in data:
            for tid in data[did]:
                count[int(tid)] += 1
        res = []
        for i, c in enumerate(count):
            if c > 100:
                res.append(str(i))
        return res

    def get_topic(line):
        """get did-topic tuples from a string"""
        topic, did, _ = line.split(' ')
        return did, topic

    def select_topics(topics):
        """select the topics containing at least 200 documents"""
        return [topic for topic in topics if len(topics[topic]) > 200]

    fp_train = r"data/RCV1/lyrl2004_vectors_train.dat"
    train = {}
    with open(fp_train, 'r') as f:
        for line in f.readlines():
            did, tids = get_tids(line.strip())
            train[did] = tids

    fp_test = r"data/RCV1/lyrl2004_vectors_test_pt0.dat"
    test = {}
    with open(fp_test, 'r') as f:
        for _ in range(30000):
            line = f.readline().strip()
            did, tids = get_tids(line.strip())
            test[did] = tids

    fp_tid = r"data/RCV1/tids.txt"
    if is_write:
        with open(fp_tid, 'w') as f:
            for tid in select_tid(train):
                f.write(tid+'\n')

    tid_used = []
    with open(fp_tid, 'r') as f:
        for line in f.readlines():
            tid_used.append(line.strip())

    fp_topic = r"data/RCV1/rcv1-v2.topics.qrels"
    topics = {}
    with open(fp_topic, 'r') as f:
        for line in f.readlines():
            did, topic = get_topic(line)
            if did in train:
                if topic in topics:
                    topics[topic].append(did)
                else:
                    topics[topic]= [did]

    fp_topic_used = r"data/RCV1/topics.txt"
    if is_write:
        with open(fp_topic_used, 'w') as f:
            for topic in select_topics(topics):
                f.write(topic+'\n')

    topic_used = []
    with open(fp_topic_used, 'r') as f:
        for line in f.readlines():
            topic_used.append(line.strip())

    fp_stem = r"data/RCV1/stem.termid.idf.map.txt"
    stem_tid = {}
    with open(fp_stem, 'r') as f:
        for line in f.readlines():
            stem, tid, _ = line.strip().split(' ')
            stem_tid[tid] = stem

    return train, test, tid_used, topics, topic_used, stem_tid


def generate_x(data, tid_used):
    """generate the design matrix and labels.
    Args:
        data: a dict whose key is a did and the corresponding
            value is a list of its topics.
        tid_used:  a list of adopted tids.
    Returns:
        the design matrix, a pd.DataFrame
    """
    X = {tid: np.zeros(len(data)) for tid in tid_used}
    index = []
    for i, did in enumerate(data):
        for tid in data[did]:
            if tid in tid_used:
                X[tid][i] = 1
        index.append(did)
    return pd.DataFrame(X, index=index)


def generate_y(index, dids, topic):
    """generate the labels.
    Args:
        index: the index of the design matrix.
        dids: a list of dids in the target topic.
        topic: the name of the target topic, a string.
    Returns:
        the vaector of labels, a pd.DataFrame
    """
    y = pd.DataFrame({topic: [0] * len(index)}, index=index)
    for did in dids:
        y[topic][did] = 1
    return y
