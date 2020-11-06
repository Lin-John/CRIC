from sklearn.metrics import precision_score, recall_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from araf import CRIC
from loaddata import load_data_rcv1, generate_x, generate_y


def rule_predict(X, rule):
    features = X.columns
    res = X
    for f, v in rule:
        res = res[res[features[f]] == v]
    pred = pd.DataFrame({'y': [0] * len(X)}, index=X.index)
    pred['y'][res.index] = 1
    return pred['y'].values.ravel()


class CRIC_tmp(CRIC):

    def __init__(self, n_freq, n_conf, max_depth=1000, max_size=4, n_chain=300,
                 binary=False, random_state=2020):
        super().__init__(n_freq, n_conf, max_depth, max_size, n_chain, binary, random_state)

    def generate_rule(self, features, target, prior):
        """Generate rules."""
        for item, conf in self.confident_rule[1].items:
            antecedents = tuple(map(lambda x: (features[x[0]], x[1]), item))
            consequence = (target, 1)
            self.rules.insert((antecedents, consequence), conf)


class RuleClassifier(object):
    """A classifier based on rules."""

    def __init__(self, n_freq, n_conf, max_depth=10000, max_size=4, n_chain=100, binary=True):
        """Init the class."""
        self.araf = CRIC_tmp(n_freq, n_conf, max_depth, max_size, n_chain, binary)

    def fit(self, X, y):
        """Fit model."""
        # print('begin fitting')
        self.araf.fit(X, pd.DataFrame({'y': y}))
        # print('end fitting')

    def predict(self, X):
        """Predict the results.
           If an instance is subject to more than a half of the confident rules,
           then it will be classified as a positive example.
        """
        # print('begin predict')
        y_pred, c_pred = {}, {}
        for rule, conf in self.araf.confident_rule[1].items:
            y_pred[rule] = rule_predict(X, rule)
            c_pred[rule] = conf
        # print('end predict')
        return y_pred, c_pred


def best_rule(y_true, y_pred):
    p_best, r_best, rule_best = 0, 0, None
    for rule in y_pred:
        y = y_pred[rule]
        if sum(y) > 0.1 * sum(y_true):
            p = precision_score(y_true, y)
            if p > p_best:
                r = recall_score(y_true, y)
                p_best, r_best, rule_best = p, r, rule
    return p_best, r_best, rule_best


def bagging_rule(y_true, y_pred, c_pred):
    y = np.zeros(len(y_true))
    conf_tt = 0
    for rule in y_pred:
        conf = c_pred[rule]
        # print(len(y), len(y_pred[rule]), conf)
        y += y_pred[rule] * conf
        conf_tt += conf
    y = np.where(y >= 0.2 * conf_tt, 1, 0)
    p = precision_score(y_true, y)
    r = recall_score(y_true, y)
    return p, r


if __name__ == "__main__":
    lr = LogisticRegression(C=1, penalty='l1', solver="liblinear", random_state=42, max_iter=1000)
    rc = RuleClassifier(n_freq=400, n_conf=200, max_depth=10000, max_size=4, n_chain=300, binary=True)

    train, test, tid_used, topics, topic_used, stem_tid = load_data_rcv1()
    X = generate_x(train, tid_used)

    n_train = 13149
    X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
    p_br, p_rs, p_rlr, p_lr = [], [], [], []
    r_br, r_rs, r_rlr, r_lr = [], [], [], []
    rule_br = []
    for topic in topic_used:
        y = generate_y(X.index, topics[topic], topic).values.ravel()
        y_train, y_test = y[:n_train], y[n_train:]

        rc.fit(X_train, y_train)
        y_pred, c_pred = rc.predict(X_test)

        lr.fit(rc.araf.transform(X_train), y_train)
        y_rlr = lr.predict(rc.araf.transform(X_test))

        lr.fit(X_train, y_train)
        y_lr = lr.predict(X_test)

        # add the results of Best_Rule
        p_, r_, rule_ = best_rule(y_test, y_pred)
        p_br.append(p_)
        r_br.append(r_)
        rule_br.append(rule_)

        # add the results of Rules
        p_, r_ = bagging_rule(y_test, y_pred, c_pred)
        p_rs.append(p_)
        r_rs.append(r_)

        # add the results of Rules_LR
        p_rlr.append(precision_score(y_test, y_rlr))
        r_rlr.append(recall_score(y_test, y_rlr))

        # add the results of LR
        p_lr.append(precision_score(y_test, y_lr))
        r_lr.append(recall_score(y_test, y_lr))

        print(topic, 'precision', p_br[-1], p_rs[-1], p_rlr[-1], p_lr[-1])
        print(topic, 'recall', r_br[-1], r_rs[-1], r_rlr[-1], r_lr[-1])
        # break

    # plot the precisions
    y = np.arange(len(topic_used))
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(y)
    ax.set_yticklabels(topic_used)
    ax.set_xlabel("precision")
    ax.set_ylabel("topic")
    plt.scatter(p_br, y, label='Best-Rule', marker='^')
    plt.scatter(p_rs, y, label='Rules', marker='x')
    plt.scatter(p_lr, y, label='LR', marker='v')
    plt.scatter(p_rlr, y, label='Rules+LR', marker='o')
    plt.grid(axis="y")
    plt.legend()
    plt.savefig('precision_rcv1.png', bbox_inches='tight')
    plt.show()

    # plot the recalls
    y = np.arange(len(topic_used))
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(y)
    ax.set_yticklabels(topic_used)
    ax.set_xlabel("precision")
    ax.set_ylabel("topic")
    plt.scatter(r_rc, y, label='Rule', marker='x')
    plt.scatter(r_lr, y, label='LR', marker='v')
    plt.scatter(r_rlr, y, label='Rule+LR', marker='o')
    plt.grid(axis="y")
    plt.legend()
    # plt.savefig('recall_rcv1.png', bbox_inches='tight')
    plt.show()
