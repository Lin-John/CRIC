from sklearn.metrics import precision_score, recall_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from ric import RIC
from loaddata import load_data_rcv1, generate_x, generate_y


def rule_predict(X, rule):
    """Classify instances by a rule."""
    features = X.columns
    res = X.copy()
    for f, v in rule:
        res = res[res[features[f]]==v]
    pred = pd.DataFrame({'y':[0]*len(X)}, index=X.index)
    pred['y'][res.index] = 1
    return pred['y'].values.ravel()


class RuleClassifier(object):
    """A classifier based on rules."""

    def __init__(self, n_freq, n_conf, max_order=2, max_depth=1000, min_size=2, n_chain=1000,
                 binary=True, positive_class=True, random_state=2020):
        """Init the class."""
        self.ric = RIC(n_freq, n_conf, max_order, max_depth, min_size, n_chain, binary, positive_class, random_state)
        self.rule = None

    def fit(self, X, y):
        p_c = sum(y)/len(y)
        self.ric.fit(X, y)
        for pattern, _ in self.ric.confident_rule[1].items[::-1]:
            if self.ric.get_frequency(pattern, 0) * (1-p_c) + self.ric.get_frequency(pattern, 1) * p_c >= p_c/10:
                self.rule = pattern
                break

    def predict(self, X):
        """Predict the results.
           If an instance is subject to more than a half of the confident rules,
           then it will be classified as a positive example.
        """
        return rule_predict(X, self.rule)



if __name__ == "__main__":
    lr = LogisticRegression(C=1, penalty='l1', solver="liblinear", random_state=42, max_iter=1000)

    train, test, tid_used, topics, topic_used, stem_tid = load_data_rcv1(is_write=True)
    X = generate_x(train, tid_used)
    features = [stem_tid[i] for i in X.columns]

    n_train = 13149
    X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]

    fp_train = r"data/RCV1/results.pkl"
    with open(fp_train, "rb") as handle:
        results = pickle.load(handle)
    # results = {}
    for topic in topic_used:
        rc = RuleClassifier(n_freq=500, n_conf=250, n_chain=1000, max_depth=100000)
        if topic in results and results[topic]["p_rule"] >= 0.4:
            continue
        print(topic)
        y = generate_y(X.index, topics[topic], topic).values.ravel()
        y_train, y_test = y[:n_train], y[n_train:]

        rc.fit(X_train, y_train)
        y_br = rc.predict(X_test)

        lr.fit(rc.ric.transform(X_train), y_train)
        y_rlr = lr.predict(rc.ric.transform(X_test))

        lr.fit(X_train, y_train)
        y_lr = lr.predict(X_test)

        results[topic] = {}
        # add the results of Best_Rule
        results[topic]['p_rule'] = precision_score(y_test, y_br)
        results[topic]['r_rule'] = recall_score(y_test, y_br)
        results[topic]['rule'] = "/".join([features[f] for f, v in rc.rule])

        # add the results of Rules_LR
        results[topic]['p_rlr'] = precision_score(y_test, y_rlr)
        results[topic]['r_rlr'] = recall_score(y_test, y_rlr)

        # add the results of LR
        results[topic]['p_lr'] = precision_score(y_test, y_lr)
        results[topic]['r_lr'] = recall_score(y_test, y_lr)

        # print(results[topic])
        # with open(fp_train, "wb") as handle:
        #     pickle.dump(results, handle)

        # fp_train = r"data/RCV1/results.pkl"
        # with open(fp_train, "rb") as handle:
        #     results = pickle.load(handle)
        # topic_used.sort(key=lambda topic: results[topic]["p_rule"])

        # plot the figure
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 12), sharey=True)
        y_label = np.arange(len(topic_used))

        ax1 = ax[0]
        ax1.set_ylim(-1, len(topic_used))
        ax1.set_yticks(y_label)
        ax1.set_yticklabels(topic_used)
        ax1.set_ylabel("topic")
        ax1.scatter([results[topic]["p_rule"] for topic in topic_used], y_label, label='Best-Rule', marker='^')
        ax1.scatter([results[topic]["p_lr"] for topic in topic_used], y_label, label='LR', marker='v')
        ax1.scatter([results[topic]["p_rlr"] for topic in topic_used], y_label, label='Rules+LR', marker='o')
        ax1.set_xlabel("precision")
        ax1.set_xlim(0, 1)
        ax1.set_xticks(np.arange(0, 1.1, 0.1))
        ax1.grid(axis="y")

        ax2 = ax[1]
        ax2.scatter([results[topic]["r_rule"] for topic in topic_used], y_label, label='Best-Rule', marker='^')
        ax2.scatter([results[topic]["r_lr"] for topic in topic_used], y_label, label='LR', marker='v')
        ax2.scatter([results[topic]["r_rlr"] for topic in topic_used], y_label, label='Rules+LR', marker='o')
        ax2.set_xlabel("recall")
        ax2.set_xlim(0, 1)
        ax2.set_xticks(np.arange(0, 1.1, 0.1))
        ax2.grid(axis="y")

        plt.legend(loc='center', bbox_to_anchor=(0, 1.02), ncol=3)
        plt.savefig("./figure/rcv1.eps", format='eps', dpi=1200)
        plt.show()