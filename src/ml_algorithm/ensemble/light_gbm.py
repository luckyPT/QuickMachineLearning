import os

os.environ["PATH"] += os.pathsep + 'C:\\Program Files\\Graphviz 2.44.1\\bin'
import lightgbm as lgb
from src.data.sk_data import Iris
from src.utils import data_split
import numpy as np

feature, label = Iris.features, Iris.label
feature = feature[label <= 1]
label = label[label <= 1]
train_feature, test_feature, train_label, test_label = data_split.split(feature, label)
train_data = lgb.Dataset(data=train_feature, label=train_label)
test_data = lgb.Dataset(data=test_feature, label=test_label)
param = {'num_leaves': 31, 'num_trees': 100, 'objective': 'binary', 'num_class': 1}
param['metric'] = 'multi_logloss'

num_round = 10
bst = lgb.train(param, train_data, num_round, valid_sets=[test_data])
bst.save_model('model.txt')
# A saved model can be loaded:
bst = lgb.Booster(model_file='model.txt')

ypred = bst.predict(test_feature, num_iteration=bst.best_iteration)
print(np.array([1 if score > 0.5 else 0 for score in ypred]))
print(test_label)

for i in range(0, num_round):
    img = lgb.create_tree_digraph(bst, tree_index=i)
    with open('trees-{}.svg'.format(i), 'w') as f:
        f.write(img._repr_svg_())
