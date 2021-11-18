import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression as LogReg
import cleanlab
from cleanlab.pruning import get_noise_indices
import src.utils.data_split as data_split

# https://github.com/cleanlab/cleanlab/blob/master/examples/mnist/label_errors_mnist_test_cnn.ipynb
# https://github.com/idealo/imagededup/issues/67
# https://blog.csdn.net/u014546828/article/details/109235539
from data.sk_data import BreastCancer


def top_mislabel_sample(features, labels, model=None):
    """
    方法的调用一定要放到main函数里面，否则可能抛出异常
    """
    jc, psx = cleanlab.latent_estimation.estimate_confident_joint_and_cv_pred_proba(features, labels,
                                                                                    LogReg(multi_class='auto',
                                                                                           solver='lbfgs') if model is None else model)
    est_py, est_nm, est_inv = cleanlab.latent_estimation.estimate_latent(jc, labels)
    noise_idx = get_noise_indices(labels, psx, est_inv, prune_method="both")
    pred = np.argmax(psx, axis=1)
    print('Number of estimated errors in test set:', sum(noise_idx))
    ordered_noise_idx = np.argsort(np.asarray([psx[i][j] for i, j in enumerate(labels)])[noise_idx])
    # 标注标签及置信度
    label4viz = labels[noise_idx][ordered_noise_idx]
    prob_given = np.asarray([psx[i][j] for i, j in enumerate(labels)])[noise_idx][ordered_noise_idx]
    # 预测标签及置信度
    pred4viz = pred[noise_idx][ordered_noise_idx]
    prob_pred = np.asarray([psx[i][j] for i, j in enumerate(pred)])[noise_idx][ordered_noise_idx]

    # 样本索引
    img_idx = np.arange(len(noise_idx))[noise_idx][ordered_noise_idx]
    return img_idx, label4viz, prob_given


if __name__ == '__main__':
    bin_feature, bin_label = BreastCancer.features, BreastCancer.label
    idx, label, prob = top_mislabel_sample(bin_feature, bin_label)
    print(idx)
    print(label)
    print(prob)
