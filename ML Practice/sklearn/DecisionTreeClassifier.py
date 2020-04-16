from sklearn import tree
import numpy as np
from sklearn.model_selection import train_test_split
import pydotplus
from io import StringIO
import os
from sklearn.metrics import roc_curve, auc
from matplotlib.legend_handler import HandlerLine2D
import matplotlib.pyplot as plt
os.environ["PATH"] += os.pathsep + 'D:\SoftWare\Anaconda3\graphviz\\bin'
data_list = []
taget = []

feature_names = ['videoNameLength','isNumberLetter','isDateStr','dlength','dflag','alength','aflag','aliasNamesNum','aliasNameslength','diffName','descriptionLength','hasNameNum','isHasName']
label = []
with open('data4modle.json') as lines:
    for line in lines:
        data_ = line.replace('\n', '').split('\t')
        data_list.append(data_[:len(data_) - 1])
        taget.append(int(data_[-1:][0]))

data = np.array(data_list)
x_train, x_test, y_train, y_test = train_test_split(data_list, taget, test_size=0.3, random_state=0)
clf = tree.DecisionTreeClassifier(criterion="gini",
                 splitter="best",
                 max_depth=8,
                 min_samples_split=0.10,
                 min_samples_leaf=0.029,
                 min_weight_fraction_leaf=0.,
                 max_features=8,
                 random_state=None,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 class_weight=None,
                 presort=False)
clf = clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)
print('baseline roc_auc is :', roc_auc)















# dot_data = StringIO()
# tree.export_graphviz(clf, out_file=dot_data,
#                      feature_names=feature_names,
#                      max_depth=5, class_names=["movie", "other"],
#                      filled=True, rounded=True, special_characters=True)
#
# #保存成 dot 文件，后面可以用 dot out.dot -T pdf -o out.pdf 转换成图片
# print("saving model")
#
# with open("out.dot", 'w') as f :
#     f = tree.export_graphviz(clf, out_file = f,
#             feature_names = feature_names)
#
#
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# print("saving pdf")
# graph.write_pdf("tree.pdf")


def adjust_depth():
    max_depths = np.linspace(1, 32, 32, endpoint=True)
    train_results = []
    test_results = []

    for max_depth in max_depths:
        dt = tree.DecisionTreeClassifier(max_depth=max_depth)
        dt.fit(x_train, y_train)
        train_pred = dt.predict(x_train)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        # Add auc score to previous train results
        train_results.append(roc_auc)
        y_pred = dt.predict(x_test)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        # Add auc score to previous test results
        test_results.append(roc_auc)
    line1, = plt.plot(max_depths, train_results, 'b', label='Train AUC')
    line2, = plt.plot(max_depths, test_results, 'r', label='Test AUC')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('AUC score')
    plt.xlabel('Tree depth')
    plt.show()

def adjust_min_split():
    min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
    train_results = []
    test_results = []
    for min_samples_split in min_samples_splits:
        dt = tree.DecisionTreeClassifier(min_samples_split=min_samples_split)
        dt.fit(x_train, y_train)
        train_pred = dt.predict(x_train)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        train_results.append(roc_auc)
        y_pred = dt.predict(x_test)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        test_results.append(roc_auc)
    line1, = plt.plot(min_samples_splits, train_results, 'b', label='Train AUC')
    line2, = plt.plot(min_samples_splits, test_results, 'r', label='Test AUC')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('AUC score')
    plt.xlabel('min samples split')
    plt.show()

def adjust_min_leaf():
    min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)
    train_results = []
    test_results = []
    for min_samples_leaf in min_samples_leafs:
        dt = tree.DecisionTreeClassifier(min_samples_leaf=min_samples_leaf)
        dt.fit(x_train, y_train)
        train_pred = dt.predict(x_train)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        train_results.append(roc_auc)
        y_pred = dt.predict(x_test)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        test_results.append(roc_auc)
    from matplotlib.legend_handler import HandlerLine2D
    line1, = plt.plot(min_samples_leafs, train_results, 'b', label='Train AUC')
    line2, = plt.plot(min_samples_leafs, test_results, 'r', label='Test AUC')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel('AUC score')
    plt.xlabel('min samples leaf')
    plt.show()
