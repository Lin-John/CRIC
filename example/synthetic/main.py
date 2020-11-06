import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def random_one_zero(size, prob):
    return np.where(np.random.uniform(size=size)<prob, 1, 0)


def generate_synthetic_data(p1, p2, p3, q1, size=10000):
    probs = [p1, p2, p3] + [0.5] * 6 + [1]
    sparse_features = [str(i+1) for i in range(len(probs))]
    dense_features = []
    target = ['y']
    data = {}
    for i, prob in enumerate(probs):
        data[str(i+1)] = random_one_zero(size, prob)
    y = np.zeros(size)
    y = np.where((data['1']==1) & (data['2']==1) & (data['3']==1), random_one_zero(size, q1), y)
    data['y'] = y.astype('int')
    return pd.DataFrame(data), dense_features, sparse_features, target, [], "binary"


def test_synthetic(generate_data, p1, p2, p3, q1, n_chains, n_freq, n_conf):
    df, dense_features, sparse_features, target, miss_val, task = generate_data(p1, p2, p3, q1)
    X, y = df[sparse_features + dense_features], df[target]

    rules = [(((0,0),), 0), (((1,0),), 0), (((2,0),), 0), (((0,1),(1,1),(2,1),), 1)]

    freqs = {x:[] for x in rules}
    confs = {x:[] for x in rules}
    for n_chain in n_chains:
        if n_chain%10 == 0:
            print(n_chain)
        araf = CRIC(n_freq, n_conf, max_size=4, n_chain=n_chain)
        araf.fit(X, y)
        item_freq = {c:{item:freq for item, freq in araf.frequent_set[c].items} for c in [0,1]}
        item_conf = {c:{item:conf for item, conf in araf.confident_rule[c].items} for c in [0,1]}
        for item, c in rules:
            if item in item_freq[c]:
                freqs[(item,c)].append(item_freq[c][item])
            else:
                freqs[(item,c)].append(0)
            if item in item_conf[c]:
                confs[(item,c)].append(item_conf[c][item])
            else:
                confs[(item,c)].append(0)
    print(araf.rules.items)
    return freqs, confs


def tostring(x):
    res = []
    item, c = x
    for index, value in item:
        res.append('X{}={}'.format(index+1, value))

    return ','.join(res)+'|y={}'.format(c)


def tostring2(x):
    res = []
    item, c = x
    for index, value in item:
        res.append('X{}={}'.format(index+1, value))

    return ','.join(res)+'->y={}'.format(c)


if __name__ == "__main__":
    load_data = generate_synthetic_data
    p1, p2, p3, q1 = 0.4, 0.5, 0.6, 0.9
    n_chains = range(1, 301)
    freqs, confs = test_synthetic(load_data, p1, p2, p3, q1, n_chains, 300, 5)

    # plot estimated frequencies
    x1, x2, x3, x4 = freqs.keys()
    color = {x1: 'y', x2: 'g', x3: 'c', x4: 'r'}
    freq_true = {x1: [(1 - p1) / (1 - p1 * p2 * p3 * q1)] * len(n_chains),
                 x2: [(1 - p2) / (1 - p1 * p2 * p3 * q1)] * len(n_chains),
                 x3: [(1 - p3) / (1 - p1 * p2 * p3 * q1)] * len(n_chains),
                 x4: [1] * len(n_chains)}

    for x in freqs:
        plt.plot(n_chains, freqs[x], color=color[x], label=tostring(x))
        plt.plot(n_chains, freq_true[x], color=color[x], linestyle=':')
    plt.xlabel("number of chains")
    plt.ylabel("estimated frequency")
    plt.legend()
    # plt.savefig('frequency.png', bbox_inches='tight')
    plt.show()

    # plot estimated confidences
    conf_true = {x1: [1] * len(n_chains), x2: [1] * len(n_chains), x3: [1] * len(n_chains), x4: [q1] * len(n_chains)}
    for x in confs:
        plt.plot(n_chains, confs[x], color=color[x], label=tostring2(x))
        plt.plot(n_chains, conf_true[x], color=color[x], linestyle=':')
    plt.xlabel("number of chains")
    plt.ylabel("estimated confidence")
    plt.legend(loc=0)
    # plt.savefig('confidence.png', bbox_inches='tight')
    plt.show()
