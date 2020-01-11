#author: mocas
# * 赛题介绍：本赛题目标是预测投放的优惠券是否核销。
# * 数据分析: 其中包含一些常见的数据处理。
# * 特征工程: 主要是简单的提取了折扣相关信息和时间信息。
import os, sys, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, auc, roc_curve
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
#首先导入数据
dfoff=pd.read_csv("/home/mocas/kaggle/o2o/data/ccf_offline_stage1_train.csv")##data for offline_train
dftest=pd.read_csv("/home/mocas/kaggle/o2o/data/ccf_offline_stage1_test_revised.csv") ##data for offline test
dfon=pd.read_csv("/home/mocas/kaggle/o2o/data/ccf_online_stage1_train.csv") ##data for online test
print("load data over")

##进行数据预处理
# dfoff.info()
# dfon.info()ccc
# 1. 将满xx减yy类型(`xx:yy`)的券变成折扣率 : `1 - yy/xx`，同时建立折扣券相关的特征 `discount_rate, discount_man, discount_jian, discount_type`
# 2. 将距离 `str` 转为 `int`
# convert Discount_rate and Distance
##以下几个函数都是对discount_rate的操作，将discount_rate分成多列操作

##chane discount to type
def getDiscountType(row):
    if pd.isnull(row):
        return np.nan
    elif ':' in row:
        return 1
    else:
        return 0

##xx:yy折扣是满xx折扣yy，则转换成折扣率是1-xx/yy
def convertRate(row):
    """Convert discount to rate"""
    if pd.isnull(row):
        return 1.0
    elif ':' in str(row):
        rows = row.split(':')
        return 1.0 - float(rows[1])/float(rows[0])
    else:
        return float(row)

##将消费金额取出分出一列
def getDiscountMan(row):
    if ':' in str(row):
        rows = row.split(':')
        return int(rows[0])
    else:
        return 0
##将减去的金额分出一列
def getDiscountJian(row):
    if ':' in str(row):
        rows = row.split(':')
        return int(rows[1])
    else:
        return 0
print("tool is ok.")

# dfoff['discount_rate2'] = getDiscountType(dfoff['Discount_rate'])

def processData(df):
    # convert discunt_rate
    df['discount_rate'] = df['Discount_rate'].apply(convertRate)
    df['discount_man'] = df['Discount_rate'].apply(getDiscountMan)
    df['discount_jian'] = df['Discount_rate'].apply(getDiscountJian)
    df['discount_type'] = df['Discount_rate'].apply(getDiscountType)
    print(df['discount_rate'].unique())
    # convert distance
    df['distance'] = df['Distance'].fillna(-1).astype(int) ##fill with -1 for fillna
    return df

dfoff = processData(dfoff) ##after process,dicount_rate change several parts,liang hua
dftest = processData(dftest) ##simalarity to shangmian查看时间范围的数值

# ###重要信息，isnull的选取问题
# a=dfoff.Distance[dfoff.Coupon_id.isnull()]
# b=dfoff.Distance.loc[dfoff.Coupon_id.isnull()]
# c=dfoff[dfoff.Coupon_id.isnull()]
# d=dfoff.loc[dfoff.Coupon_id.isnull(),'Coupon_id']
# ####
date_received = dfoff['Date_received'].unique() ##查看收到优惠券时间范围的数值
date_received = sorted(date_received[pd.notnull(date_received)]) ##将收到优惠券时间范围排序
# print(date_received)
date_buy = dfoff['Date'].unique() ##查看日期数值
date_buy = sorted(date_buy[pd.notnull(date_buy)])##将消费日期排序
date_buy = sorted(dfoff[dfoff['Date'].notnull()]['Date']) #显示所有非空消费日期数据
couponbydate = dfoff[dfoff['Date_received'].notnull()][['Date_received', 'Date']].groupby(['Date_received'], as_index=False).count()
couponbydate.columns = ['Date_received','count'] ##修改行列的索引index
buybydate = dfoff[(dfoff['Date'].notnull()) & (dfoff['Date_received'].notnull())][['Date_received', 'Date']].groupby(['Date_received'], as_index=False).count()
buybydate.columns = ['Date_received','count']
print("end")


def getWeekday(row):
    if row == 'nan':
        return np.nan
    else:
        return date(int(row[0:4]), int(row[4:6]), int(row[6:8])).weekday() + 1


dfoff['weekday'] = dfoff['Date_received'].astype(str).apply(getWeekday)
dftest['weekday'] = dftest['Date_received'].astype(str).apply(getWeekday)

# weekday_type :  周六和周日为1，其他为0
dfoff['weekday_type'] = dfoff['weekday'].apply(lambda x : 1 if x in [6,7] else 0)
dftest['weekday_type'] = dftest['weekday'].apply(lambda x : 1 if x in [6,7] else 0)

# change weekday to one-hot encoding，将weekday转换成one-hot编码
weekdaycols = ['weekday_' + str(i) for i in range(1,8)]
tmpdf = pd.get_dummies(dfoff['weekday'].replace('nan', np.nan)) ##如果是nan，则one——hot为0
tmpdf.columns = weekdaycols
dfoff[weekdaycols] = tmpdf

tmpdf = pd.get_dummies(dftest['weekday'].replace('nan', np.nan))
tmpdf.columns = weekdaycols
dftest[weekdaycols] = tmpdf


def label(row):
    if pd.isnull(row['Date_received']):
        return -1
    if pd.notnull(row['Date']):
        td = pd.to_datetime(row['Date'], format='%Y%m%d') -  pd.to_datetime(row['Date_received'], format='%Y%m%d')
        if td <= pd.Timedelta(15, 'D'):
            return 1
    return 0

dfoff['label'] = dfoff.apply(label, axis = 1)

# data split
print("-----data split------")
df = dfoff[dfoff['label'] != -1].copy()
train = df[(df['Date_received'] < 20160516)].copy() ##查找满足日期小于2016.5.16
valid = df[(df['Date_received'] >= 20160516) & (df['Date_received'] <= 20160615)].copy()
print("end")


# feature
def check_model(data, predictors):
    classifier = lambda: SGDClassifier(
        loss='log',  # loss function: logistic regression
        penalty='elasticnet',  # L1 & L2
        fit_intercept=True,  # 是否存在截距，默认存在
        max_iter=100,
        shuffle=True,  # Whether or not the training data should be shuffled after each epoch
        n_jobs=1,  # The number of processors to use
        class_weight=None)  # Weights associated with classes. If not given, all classes are supposed to have weight one.

    # 管道机制使得参数集在新数据集（比如测试集）上的重复使用，管道机制实现了对全部步骤的流式化封装和管理。
    model = Pipeline(steps=[
        ('ss', StandardScaler()),  # transformer
        ('en', classifier())  # estimator
    ])

    parameters = {
        'en__alpha': [0.001, 0.01, 0.1],
        'en__l1_ratio': [0.001, 0.01, 0.1]
    }

    # StratifiedKFold用法类似Kfold，但是他是分层采样，确保训练集，测试集中各类别样本的比例与原始数据集中相同。
    folder = StratifiedKFold(n_splits=3, shuffle=True)

    # Exhaustive search over specified parameter values for an estimator.
    grid_search = GridSearchCV(
        model,
        parameters,
        cv=folder,
        n_jobs=-1,  # -1 means using all processors
        verbose=1)
    grid_search = grid_search.fit(data[predictors],
                                  data['label'])

    return grid_search


# model.fit(train[original_feature], train['label'])
# # #### 预测以及结果评价
# print(model.score(valid[original_feature], valid['label']))

##b保存参数
# print("---save model---")
# with open('1_model.pkl', 'wb') as f:
#     pickle.dump(model, f)
# with open('1_model.pkl', 'rb') as f:
#     model = pickle.load(f)
# feature
original_feature = ['discount_rate','discount_type','discount_man', 'discount_jian','distance', 'weekday', 'weekday_type'] + weekdaycols
predictors = original_feature
model = check_model(train, predictors)

# valid predict
y_valid_pred = model.predict_proba(valid[predictors])
valid1 = valid.copy()
valid1['pred_prob'] = y_valid_pred[:, 1]
valid1.head(5)

# test prediction for submission
y_test_pred = model.predict_proba(dftest[original_feature])
dftest1 = dftest[['User_id','Coupon_id','Date_received']].copy()
dftest1['label'] = y_test_pred[:,1]
dftest1.to_csv('submit1.csv', index=False, header=False)
dftest1.head()

# avgAUC calculation
vg = valid1.groupby(['Coupon_id'])
aucs = []
for i in vg:
   tmpdf = i[1]
   if len(tmpdf['label'].unique()) != 2:
       continue
   fpr, tpr, thresholds = roc_curve(tmpdf['label'], tmpdf['pred_prob'], pos_label=1)
   aucs.append(auc(fpr, tpr))
print(np.average(aucs))
# test prediction for submission
y_test_pred = model.predict_proba(dftest[predictors])
dftest1 = dftest[['User_id','Coupon_id','Date_received']].copy()
dftest1['Probability'] = y_test_pred[:,1]
dftest1.to_csv('submit.csv', index=False, header=False)
dftest1.head(5)
