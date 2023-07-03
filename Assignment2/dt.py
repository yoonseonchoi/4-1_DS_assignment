import sys
import pandas as pd
import numpy as np
import math
from collections import Counter

# trian dataset 불러오기
train = sys.argv[1]
train_dataset = pd.read_csv(train, sep = "\t", engine='python', encoding = "cp949")
x_train = train_dataset.iloc[:, :-1].values
y_train = train_dataset.iloc[:, -1].values

# feature selection을 위한 gain ratio 계산
def gainRatio(train_dataset, f_idx):
    f_vals = train_dataset.iloc[:, f_idx].values
    f_val_cnt = {}
    for f_val in f_vals:
        if f_val not in f_val_cnt:
            f_val_cnt[f_val] = 1
        else:
            f_val_cnt[f_val] += 1

    infoD = 0
    class_vals = train_dataset.iloc[:, -1].values
    class_val_cnt = {}
    for class_val in class_vals:
        if class_val not in class_val_cnt:
            class_val_cnt[class_val] = 1
        else:
            class_val_cnt[class_val] += 1
    for cnt in class_val_cnt.values():
        p = cnt / len(class_vals)
        infoD += -p*math.log2(p)

    infoAD = 0
    for f_val, cnt in f_val_cnt.items():
        prob = cnt / len(f_vals)
        f_val_data = train_dataset[f_vals == f_val]
        f_val_class = f_val_data.iloc[:, -1].values
        f_val_class_cnt = {}
        for class_val in f_val_class:
            if class_val not in f_val_class_cnt:
                f_val_class_cnt[class_val] = 1
            else:
                f_val_class_cnt[class_val] += 1
        for class_val_cnt in f_val_class_cnt.values():
            p = class_val_cnt / len(f_val_class)
            infoAD += prob*(-p*math.log2(p))
        gain = infoD-infoAD

    splitInfo = 0
    for cnt in f_val_cnt.values():
        prob = cnt / len(f_vals)
        splitInfo += -prob*math.log2(prob)
    
    gain_ratio = gain / splitInfo
    return gain_ratio

# tree 구축을 위한 Node 초기화
class Node:
    def __init__(self, f_idx=None, threshold=None, left=None, right=None, label=None):
        self.f_idx = f_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label

# decision tree 구현
class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
    
    # train
    def fit(self, x, y):
        self.num_classes = len(set(y))
        self.num_features = x.shape[1]
        self.tree = self.growTree(x, y)
    
    # class 예측
    def predict(self, x):
        return [self.traverseTree(i, self.tree) for i in x]
    
    # tree 구축
    def growTree(self, x, y, depth=0):
        num_trains, num_features = x.shape
        num_classes = len(set(y))
        if (self.max_depth != None and depth >= self.max_depth) or (num_trains < self.min_samples_split) or num_classes == 1:
             label = self.mostCommonLabel(y)
             return Node(label=label)
        best_gain_ratio = -1
        best_f_idx = None
        best_threshold = None
        for f_idx in range(num_features):
            col_vals = x[:, f_idx]
            for threshold in np.unique(col_vals):
                left_idx = col_vals < threshold
                right_idx = ~left_idx
                if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
                    continue
                gain_ratio = gainRatio(train_dataset, f_idx)
                if gain_ratio > best_gain_ratio:
                    best_gain_ratio = gain_ratio
                    best_f_idx = f_idx
                    best_threshold = threshold
        left_idx = x[:, best_f_idx] < best_threshold
        right_idx = ~left_idx
        left = self.growTree(x[left_idx, :], y[left_idx], depth+1)
        right = self.growTree(x[right_idx, :], y[right_idx], depth+1)
        return Node(best_f_idx, best_threshold, left, right)
    
    # tree traverasal
    def traverseTree(self, x, node):
        if node.label != None:
            return node.label
        if x[node.f_idx] < node.threshold:
            return self.traverseTree(x, node.left)
        else:
            return self.traverseTree(x, node.right)
    
    def mostCommonLabel(self, y):
        cnt = Counter(y)
        common = cnt.most_common(1)[0][0]
        return common

if __name__ == "__main__":
    # 올바르게 compile하지 않으면 다음과 같은 경고가 뜸
    if len(sys.argv) != 4:
        raise Exception("Correct usage: [program] [train] [test] [result]")

    # test dataset과 result file 불러오기
    test = sys.argv[2]
    result = sys.argv[3]

    test_dataset = pd.read_csv(test, sep = "\t", engine='python', encoding = "cp949")
    x_test = test_dataset.values

    # decision tree train 및 prediction
    dt = DecisionTree()
    dt.fit(x_train, y_train)
    y_pred = dt.predict(x_test)

    result_dataset = test_dataset.copy()
    result_dataset[train_dataset.columns[-1]] = y_pred
    
    with open(result, 'w') as file:
        result_dataset.to_csv(file, sep='\t', index=False)