import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from xgboost import plot_tree
from sklearn.utils import shuffle
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score,accuracy_score,roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from itertools import cycle

from xgboost import plot_importance
from matplotlib import pyplot

plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


import xgboost as xgb




data = pd.read_csv('train_magnetic.csv')
data = data.copy()

data = shuffle(data)
label = list(data['is_magnetic'])
    two_class = []

three_label = list(data['thermodynamic_stability_level'])
three_label = label_binarize(three_label, classes=[1, 2, 3])
n_classes = 3
X = data.drop(['is_magnetic', 'thermodynamic_stability_level', 'materials'], axis=1)
mm = MinMaxScaler()
X = mm.fit_transform(X)
x_train = X[0:3000]
x_test = X[3000:]
y_train = three_label[0:3000]
y_test = three_label[3000:]

kfold = StratifiedKFold(n_splits = 5 , shuffle = True , random_state = 3)

def modelfit(alg , X_train , y_train , cv_folds = None , early_stopping_rounds = 10):
    xgb_param = alg.get_xgb_params()
    xgb_param['num_class'] = 3
     
    xgtrain = xgb.DMatrix(X_train , label = y_train)
     
    cvresult = xgb.cv(xgb_param , xgtrain , num_boost_round = alg.get_params()['n_estimators'] , folds = cv_folds ,metrics='mlogloss' , early_stopping_rounds = early_stopping_rounds )
             
         
    cvresult.to_csv('l_nestimators.csv' , index_label = 'n_estimators')

    n_estimators = cvresult.shape[0]
    print("n_estimators :")
    print(n_estimators)
     
    alg.set_params(n_estimators = n_estimators)
    alg.fit(X_train , y_train , eval_metric = 'mlogloss')
     
    train_predprob = alg.predict_proba(X_train)
    logloss = log_loss(y_train , train_predprob)
     
    print("logloss of train :")
    print (logloss)
    
xgb1 = XGBClassifier(
        learning_rate = 0.15,
        n_estimators = 500,
        max_depth = 5 ,
        min_child_weight = 1,
        gamma = 0,
        subsample = 0.3,
        colsample_bytree = 0.8,
        colsample_bylevel = 0.7,
        objective = 'multi:softprob',
        seed = 3)
 
modelfit(xgb1 , X_train , y_train , cv_folds = kfold)[object Object]

