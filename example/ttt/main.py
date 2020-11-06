import pandas as pd
from araf import CRIC
import matplotlib.pyplot as plt


if __name__ == "__main__":
    fp = r"data/ttt/tic-tac-toe.data"
    data = pd.read_csv(fp, names=['a1', 'a2', 'a3', 'b1', 'b2', 'b3', 'c1', 'c2', 'c3', 'win'])
    X, Y = data.iloc[:, :9], data.iloc[:, [9]]
    n_train = 500
    X_train, Y_train = X[:n_train], Y[:n_train]
    X_test, Y_test = X[n_train:], Y[n_train:]
    valid_feature = {'o': {'a1=o,a2=o,a3=o', 'b1=o,b2=o,b3=o', 'c1=o,c2=o,c3=o', 'a1=o,b1=o,c1=o', 'a2=o,b2=o,c2=o',
                           'a3=o,b3=o,c3=o', 'a1=o,b2=o,c3=o', 'a3=o,b2=o,c1=o'},
                     'x': {'a1=x,a2=x,a3=x', 'b1=x,b2=x,b3=x', 'c1=x,c2=x,c3=x', 'a1=x,b1=x,c1=x', 'a2=x,b2=x,c2=x',
                           'a3=x,b3=x,c3=x', 'a1=x,b2=x,c3=x', 'a3=x,b2=x,c1=x'}}

    n_freqs = range(20, 1001, 2)
    valid = {'x': [], 'o': []}
    for n_freq in n_freqs:
        if n_freq % 50 == 0:
            print(n_freq)
        araf = CRIC(n_freq=n_freq, n_conf=20, max_depth=10000, max_size=4, n_chain=1000)
        araf.fit(X, Y)
        valid['x'].append(len(set(araf.new_features) & valid_feature['x']))
        valid['o'].append(len(set(araf.new_features) & valid_feature['o']))

    # plot the number of found valid rules
    plt.figure(figsize=(10, 5))
    plt.plot(n_freqs, valid['x'], label='positive')
    plt.plot(n_freqs, valid['o'], label='negative')
    plt.xlabel("number of frequent sets")
    plt.ylabel("number of valid rules")
    plt.legend()
    plt.savefig('valid_rule.png', bbox_inches='tight')
    plt.show()
