import pandas as pd
import numpy as np


def load_data_adult():
    # load data set: adult
    cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
            'marital-status', 'occupation', 'relationship',
            'race', 'sex', 'capital-gain', 'capital-loss',
            'hours-per-week', 'native-country', 'label']
    dense_features = ['age', 'fnlwgt', 'education-num',
                      'capital-gain', 'capital-loss', 'hours-per-week']
    sparse_features = ['workclass', 'education', 'marital-status',
                       'occupation', 'relationship', 'race',
                       'sex', 'native-country']
    target = ['label']
    miss_val = [' ?']
    task = 'binary'

    dfTrain = pd.read_csv(r"data/adult/adult.data", names=cols)
    dfTest = pd.read_csv(r"data/adult/adult.test", names=cols).loc[1:].reset_index(drop=True)

    dfTest.loc[dfTest['label'] == ' <=50K.', 'label'] = ' <=50K'
    dfTest.loc[dfTest['label'] == ' >50K.', 'label'] = ' >50K'
    df = pd.concat((dfTrain, dfTest), axis=0).reset_index(drop=True)
    return df, dense_features, sparse_features, target, miss_val, task


def load_data_hd():
    # load data set: Heart Disease
    cols = ['age', 'sex', 'cp', 'trestbps', 'chol',
            'fbs', 'restecg', 'thalach',
            'exang', 'oldpeak', 'slope', 'ca',
            'thal', 'num']
    dense_features = ['trestbps', 'chol', 'thalach', 'oldpeak']
    sparse_features = ['age', 'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    target = ['num']
    miss_val = ['?']
    task = 'binary'

    df = pd.read_csv(r"data/hd/processed.cleveland.data", names=cols)
    df.loc[df['num'] != 0, 'num'] = 1
    return df, dense_features, sparse_features, target, miss_val, task


def load_data_dccc():
    # load data set: Default of Credit Card Clients
    filename = r"data/dccd/default of credit card clients.xls"
    sparse_features = ['X{}'.format(i) for i in range(2, 12)]
    dense_features = ['X1'] + ['X{}'.format(i) for i in range(12, 24)]
    target = ['Y']
    miss_val = []
    task = 'binary'
    df = pd.read_excel(filename, index_col=0)[1:].reset_index(drop=True)
    df[target] = df[target].astype('int')
    return df, dense_features, sparse_features, target, miss_val, task


def load_data_ce():
    # load data set: Car Evaluation
    filename = r"data/ce/car.txt"
    cols = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']
    dense_features = []
    sparse_features = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
    target = ['label']
    miss_val = []
    task = 'muliticlass'
    with open(filename, 'r') as f:
        df = pd.read_csv(filename, names=cols)
    return df, dense_features, sparse_features, target, miss_val, task


def load_data_lg():
    # load data set: Laigang
    # train: 0-599, valid: 600-699, test: 700:
    filename = r"data/bf/莱钢特征选择.xlsx"
    df = pd.read_excel(filename)
    sparse_features = []
    dense_features = ['fs', 'pc', 'bv', 'bt', 'gp', '上一炉Si', '上一炉S',
                      'fs(t-1)', 'pc(t-1)', 'bv(t-1)', 'bt(t-1)', 'gp(t-1)', 'fs(t-2)',
                      'pc(t-2)', 'fq(t-2)', 'bt(t-2)', 'gp(t-2)', 'fs(t-3)', 'pc(t-3)',
                      'bv(t-3)', 'bt(t-3)', 'gp(t-3)', 'fs(t-4)', 'pc(t-4)', 'bv(t-4)',
                      'bt(t-4)', 'gp(t-4)']
    target = ['label']
    miss_val = []
    task = 'multiclass'
    df['label'] = np.zeros(len(df))
    # df.loc[df['si']<0.3736, 'label'] = -1
    # df.loc[df['si']>0.8059, 'label'] = 1
    df.loc[df['si'] >= 0.3736, 'label'] = 1
    df.loc[df['si'] > 0.8059, 'label'] = 2
    X_train, X_test = df.loc[:699, dense_features], df.loc[700:, dense_features]
    y_train, y_test = df.loc[:699, target], df.loc[700:, target]
    return X_train, X_test, y_train, y_test, dense_features, sparse_features, target, miss_val, task


def load_data_bg(begin=5):
    # train: 0-599, valid: 600-699, test: 700:
    def get_set(index, df, dense_features):
        X = {}
        for f in dense_features:
            if f not in ["SI", "S"]:
                X[f] = df.loc[index, f].values
                for i in range(1, 6):
                    X["{}(t-{})".format(f, i)] = df.loc[index - i, f].values
            else:
                X[f + "(t-1)"] = df.loc[index - 1, f].values
        return pd.DataFrame(X)

    filename = r"data/bf/包钢2007-8-13采集数据.xls"
    df = pd.read_excel(filename, sheet_name="20变量")
    dense_features = ['SI', 'S', '风量', '风温', '风压', '顶压',
                      '料速', '透气性', '喷煤', '富氧量', '顶温', 'CO',
                      'CO2', 'H2', '配料碱度', '配料焦炭负荷']
    sparse_features = []
    target = ["label"]
    miss_val = []
    task = 'multiclass'
    df['label'] = np.zeros(len(df))
    train_index = np.arange(begin, begin + 700)
    test_index = np.arange(begin + 700, begin + 800)
    df.loc[df['SI'] >= 0.4132, 'label'] = 1
    df.loc[df['SI'] > 0.8251, 'label'] = 2
    X_train = get_set(train_index, df, dense_features)
    X_test = get_set(test_index, df, dense_features)
    y_train = df.loc[train_index, target].reset_index(drop=True)
    y_test = df.loc[test_index, target].reset_index(drop=True)
    return X_train, X_test, y_train, y_test, list(X_train.columns), sparse_features, target, miss_val, task
