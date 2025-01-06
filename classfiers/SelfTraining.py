#!/usr/bin/env python
# -*- coding: utf-8 -*-


# =========================================================================================================
# ================================ 0. 模块


import numpy as np
import math
from numpy import linalg

import sklearn
from sklearn import datasets
from sklearn.neighbors import kneighbors_graph

import scipy.optimize as sco

from itertools import cycle, islice


# =========================================================================================================
# ================================ 1. 算法


class SelfTraining(object):

    def __init__(self, classifier, max_iter, score_min=0.8):
        """
        自训练算法

        参数
        ----------
        classifier : sklearn分类器
            基学习器
        max_iter : 整数
            最大迭代次数
        score_min: 浮点数 [0, 1]
            在第k次迭代中，仅保留置信度高于score_min的标签
        """
        self.classifier = classifier
        self.max_iter = max_iter
        self.score_min = score_min

    def fit(self, X, X_no_label, Y):
        """
        训练模型

        参数
        ----------
        X : ndarray 形状 (n_labeled_samples, n_features)
            有标签的数据
        X_no_label : ndarray 形状 (n_unlabeled_samples, n_features)
            无标签的数据
        Y : ndarray 形状 (n_labeled_samples,)
            标签
        """
        iter_no = 0
        self.final_classifier = self.classifier

        if self.max_iter == 1:
            print('仅启动监督学习', end='...')
        else:
            print('启动自训练算法 ...\n')
            print(' | '.join([name.center(11) for name in ["迭代次数", "有标签样本数"]]))

        while (X_no_label.shape[0] > 0) & (iter_no < self.max_iter):

            # 在Sk上训练分类器
            self.final_classifier = self.final_classifier.fit(X, Y)

            # 获取无标签数据的置信分数
            scores = self.final_classifier.predict_proba(X_no_label).max(axis=1)
            ix = np.isin(range(len(scores)), np.where(scores > self.score_min)[0])

            # 保留置信度最高的样本并更新数据
            best_pred = self.final_classifier.predict(X_no_label)[ix]
            X = np.concatenate([X, X_no_label[ix]], axis=0)
            Y = np.concatenate([Y, best_pred])
            X_no_label = X_no_label[~ix]

            # 计算距离并增加一次迭代
            iter_no += 1

            # 监控进度
            if self.max_iter == 1:
                print('完成')
            else:
                print(' | '.join([("%d" % iter_no).rjust(11), ("%d" % sum(ix)).rjust(11)]))

    def predict(self, X):
        """
        自训练算法

        参数
        ----------
        X : ndarray 形状 (n_samples, n_features)


        返回值
        ---------
        predictions: ndarray 形状 (n_samples, )
        """
        return self.final_classifier.predict(X)

    def accuracy(self, Xtest, Ytrue):
        """
        参数
        ----------
        Xtest : ndarray 形状 (n_samples, n_features)
            测试数据
        Ytrue : ndarray 形状 (n_samples, )
            测试标签
        """
        predictions = self.predict(Xtest)
        accuracy = sum(predictions == Ytrue) / len(predictions)
        print('准确率: {}%'.format(round(accuracy * 100, 2)))
