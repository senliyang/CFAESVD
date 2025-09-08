import numpy as np
seed = 2024
np.random.seed(seed)


# 组合
def generate_f1(D,train_samples,feature_m,d_data1,feature_MFm, feature_MFd):
    seed = 2024
    np.random.seed(seed)
    vect_len1 = feature_m.shape[1]
    vect_len2 = d_data1.shape[1]
    train_n = train_samples.shape[0]
    train_feature = np.zeros([train_n, 2 * vect_len1 + 2 * D])
    # train_feature = np.zeros([train_n, vect_len1+vect_len2])
    train_label = np.zeros([train_n])
    for i in range(train_n):
        train_feature[i, 0:vect_len1] = feature_m[train_samples[i, 0], :]
        train_feature[i, vect_len1:(vect_len1 + vect_len2)] = d_data1[134*train_samples[i, 0]+train_samples[i, 1], :]
        train_feature[i, (vect_len1 + vect_len2):(vect_len1 + vect_len2 + D)] = feature_MFm[train_samples[i, 0], :]
        train_feature[i, (vect_len1 + vect_len2 + D):(vect_len1 + vect_len2 + 2 * D)] = feature_MFd[train_samples[i, 1],:]
        train_label[i] = train_samples[i, 2]
    return train_feature, train_label
