import numpy as np
def precision_recall_f1(y_true, y_pred):
    tp = sum((yt == 1 and yp == 1) for yt, yp in zip(y_true, y_pred))
    fp = sum((yt == 0 and yp == 1) for yt, yp in zip(y_true, y_pred))
    fn = sum((yt == 1 and yp == 0) for yt, yp in zip(y_true, y_pred))

    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return precision, recall, f1


# 测试数据
y_true = [0, 1, 1, 1, 0, 1, 0, 0, 1, 0]
y_pred = [0, 1, 1, 0, 0, 1, 0, 1, 1, 0]

precision, recall, f1 = precision_recall_f1(y_true, y_pred)
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


def AUC(label, pre):
    pos = [i for i in range(len(label)) if label[i] == 1]
    neg = [i for i in range(len(label)) if label[i] == 0]

    auc = 0
    for i in pos:
        for j in neg:
            if pre[i] > pre[j]:
                auc += 1
            elif pre[i] == pre[j]:
                auc += 0.5

    return auc / (len(pos) * len(neg))





y = np.array([1, 0, 0, 0, 1, 0, 1, 0, ])
pred = np.array([0.9, 0.8, 0.3, 0.1, 0.4, 0.9, 0.66, 0.7])

fpr, tpr, thresholds = roc_curve(y, pred, pos_label=1)
print("-----sklearn:", auc(fpr, tpr))
print("-----py脚本:", AUC(y, pred))


def dcg_at_k(r, k):
    r = np.array(r)[:k]
    dcg = sum((2 ** rel - 1) / np.log2(idx + 2) for idx, rel in enumerate(r))
    return dcg


def ndcg_at_k(y_true, y_scores, k=10):
    sorted_indices = sorted(range(len(y_scores)), key=lambda i: y_scores[i], reverse=True)
    y_true_sorted = [y_true[i] for i in sorted_indices[:k]]

    dcg = dcg_at_k(y_true_sorted, k)
    idcg = dcg_at_k(sorted(y_true, reverse=True), k)

    return dcg / idcg if idcg > 0 else 0


# 测试数据
y_true = [3, 2, 3, 0, 1, 2]
y_scores = [0.9, 0.7, 0.8, 0.3, 0.2, 0.4]

ndcg = ndcg_at_k(y_true, y_scores, k=5)
print(f"NDCG: {ndcg}")
