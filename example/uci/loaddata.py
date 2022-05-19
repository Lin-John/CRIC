import pandas as pd
import numpy as np
from os.path import join


def load_data_adult(path=''):
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

    dfTrain = pd.read_csv(join(path, r"data/adult/adult.data"), names=cols)
    dfTest = pd.read_csv(join(path, r"data/adult/adult.test"), names=cols).loc[1:].reset_index(drop=True)

    dfTest.loc[dfTest['label'] == ' <=50K.', 'label'] = ' <=50K'
    dfTest.loc[dfTest['label'] == ' >50K.', 'label'] = ' >50K'
    df = pd.concat((dfTrain, dfTest), axis=0).reset_index(drop=True)
    return df, dense_features, sparse_features, target, miss_val, task


def load_data_hd(path=''):
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

    df = pd.read_csv(join(path, r"data/hd/processed.cleveland.data"), names=cols)
    df.loc[df['num'] != 0, 'num'] = 1
    return df, dense_features, sparse_features, target, miss_val, task


def load_data_dccc(path=''):
    # load data set: Default of Credit Card Clients
    filename = join(path, r"data/dccd/default of credit card clients.xls")
    sparse_features = ['X{}'.format(i) for i in range(2, 12)]
    dense_features = ['X1'] + ['X{}'.format(i) for i in range(12, 24)]
    target = ['Y']
    miss_val = []
    task = 'binary'
    df = pd.read_excel(filename, index_col=0)[1:].reset_index(drop=True)
    df[target] = df[target].astype('int')
    return df, dense_features, sparse_features, target, miss_val, task


def load_data_ce(path=''):
    # load data set: Car Evaluation
    filename = join(path, r"data/ce/car.txt")
    cols = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']
    dense_features = []
    sparse_features = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
    target = ['label']
    miss_val = []
    task = 'muliticlass'
    with open(filename, 'r') as f:
        df = pd.read_csv(filename, names=cols)
    return df, dense_features, sparse_features, target, miss_val, task


def load_data_ccu(path=''):
    fp_name = join(path, r"data/ccu/names.txt")
    names = []
    with open(fp_name, 'r') as f:
        for line in f.readlines():
            names.append(line.strip())
    df = pd.read_csv(r"data/ccu/CommViolPredUnnormalizedData.txt", names=names)
    df=df[~df['ViolentCrimesPerPop'].isin(['?'])]
    for f in ['murders', 'rapes', 'robberies', 'assaults']:
        df.loc[:,f] = df[f].astype('int')
    df.loc[:,'ViolentCrimesPerPop'] = df['ViolentCrimesPerPop'].astype('float')
    df = df[df['murders']+df['rapes']+df['robberies']+df['assaults'] < 1.25/100000*df['ViolentCrimesPerPop']*df['population']]
    df = df[df['murders']+df['rapes']+df['robberies']+df['assaults'] > 0.8/100000*df['ViolentCrimesPerPop']*df['population']]
    droped = ['state', 'communityname', 'countyCode', 'communityCode', 'fold',
              'murders', 'murdPerPop', 'rapes', 'rapesPerPop', 'robberies',
              'robbbPerPop', 'assaults', 'assaultPerPop', 'burglaries', 'burglPerPop',
              'larcenies', 'larcPerPop', 'autoTheft', 'autoTheftPerPop', 'arsons',
              'arsonsPerPop', 'nonViolPerPop']
    columns = []
    df = df.drop(droped, axis=1)
    for f in df.columns:
        if not any(df[f].isin(['?'])):
            columns.append(f)
    return df[columns], columns[:-1], [], columns[-1:], [], 'regression'


def load_data_isolet(path=''):
    # load data set: Heart Disease
    cols = ['f' + str(i) for i in range(617)] + ['label']
    dense_features = ['f' + str(i) for i in range(617)]
    sparse_features = []
    target = ['label']
    miss_val = ['?']
    task = 'multiclass'

    E_set = [ord(c) - ord('A') + 1 for c in ['B', 'C', 'D', 'E', 'G', 'P', 'T', 'V', 'Z']]
    df1 = pd.read_csv(join(path, r"data/ISOLET/isolet1+2+3+4.data"), names=cols)
    df1 = df1[df1['label'].isin(E_set)]
    df2 = pd.read_csv(join(path, r"data/ISOLET/isolet5.data"), names=cols)
    df2 = df2[df2['label'].isin(E_set)]

    return pd.concat((df1, df2), axis=0), dense_features, sparse_features, target, miss_val, task
