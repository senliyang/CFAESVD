import warnings
import itertools
from samples import *
from four_AE import *
from GAE_trainer import *
from GAE import *
from generate_f1 import *
from numpy import interp
from metric import *
import numpy as np
from cvxopt import solvers, matrix
from sklearn import metrics
from cvxopt  import matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold
from deepforest import CascadeForestClassifier
from sklearn.metrics import roc_curve,auc,precision_recall_curve
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.metrics import auc,precision_score, recall_score, f1_score,accuracy_score
warnings.filterwarnings("ignore")


# parameter
n_splits = 5
classifier_epochs = 50
m_threshold = [0.7]
epochs=[200]
fold = 0
result = np.zeros((1, 7), float)
tprs=[]
aucs=[]
mean_fpr=np.linspace(0,1,100)

#cosine similarity
def cos_sim(MD):
    m = MD.shape[0]
    n = MD.shape[1]
    cos_MS1 = []
    cos_DS1 = []
    for i in range(m):
        for j in range(m):
            a = MD[i, :]
            b = MD[j, :]
            a_norm = np.linalg.norm(a)
            b_norm = np.linalg.norm(b)
            if a_norm != 0 and b_norm != 0:
                cos_ms = np.dot(a, b) / (a_norm * b_norm)
                cos_MS1.append(cos_ms)
            else:
                cos_MS1.append(0)

    for i in range(n):
        for j in range(n):
            a1 = MD[:, i]
            b1 = MD[:, j]
            a1_norm = np.linalg.norm(a1)
            b1_norm = np.linalg.norm(b1)
            if a1_norm != 0 and b1_norm != 0:
                cos_ds = np.dot(a1, b1) / (a1_norm * b1_norm)
                cos_DS1.append(cos_ds)
            else:
                cos_DS1.append(0)

    cos_MS1 = np.array(cos_MS1).reshape(m, m)
    cos_DS1 = np.array(cos_DS1).reshape(n, n)
    return cos_MS1, cos_DS1
#Gaussian interaction profile kernel similarity
def r_func(MD):
    m = MD.shape[0]
    n = MD.shape[1]
    EUC_MD = np.linalg.norm(MD, ord=2, axis=1, keepdims=False)
    EUC_DL = np.linalg.norm(MD.T, ord=2, axis=1, keepdims=False)
    EUC_MD = EUC_MD ** 2
    EUC_DL = EUC_DL ** 2
    sum_EUC_MD = np.sum(EUC_MD)
    sum_EUC_DL = np.sum(EUC_DL)
    rl = 1 / ((1 / m) * sum_EUC_MD)
    rt = 1 / ((1 / n) * sum_EUC_DL)
    return rl, rt
def Gau_sim(MD, rl, rt):
    MD = np.mat(MD)
    DL = MD.T
    m = MD.shape[0]
    n = MD.shape[1]
    c = []
    d = []
    for i in range(m):
        for j in range(m):
            b_1 = MD[i] - MD[j]
            b_norm1 = np.linalg.norm(b_1, ord=None, axis=1, keepdims=False)
            b1 = b_norm1 ** 2
            b1 = math.exp(-rl * b1)
            c.append(b1)
    for i in range(n):
        for j in range(n):
            b_2 = DL[i] - DL[j]
            b_norm2 = np.linalg.norm(b_2, ord=None, axis=1, keepdims=False)
            b2 = b_norm2 ** 2
            b2 = math.exp(-rt * b2)
            d.append(b2)
    GMM = np.mat(c).reshape(m, m)
    GDD = np.mat(d).reshape(n, n)
    return GMM, GDD

def get_CKA_Wi(P, q, G, h, A, b):
    '''
    数值如下：除了b都是matrix'
    l = 6
    P = get_P(train_x_k_list , gamma_list)
    q = get_q(train_x_k_list , train_y)
    G = np.identity(l)
    h = np.zeros([l,1])
    A = np.ones([1,l])
    b=matrix(1.)
    '''
    P = matrix(P)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)
    A = matrix(A)
    sol = solvers.qp(P, q, G, h, A, b)
    print(sol['x'])
    return (sol['x'])

def get_trace(a, b):
    '''
    计算<a , b>F
    Trace(a.T*b)
    '''
    return np.trace(np.dot(a.T, b))

def get_P(train_x_k_list):
    '''
    获得P矩阵
    input:train_x_list 训练集高斯矩阵 集合
    output: P矩阵
    '''
    l = len(train_x_k_list)
    n = len(train_x_k_list[0])  # n行
    # 计算Un Un = In - (1/n)ln*ln.T
    In = np.identity(n)
    ln = np.ones([n, 1])
    Un = In - (1 / n) * np.dot(ln, ln.T)
    # 计算P
    P = np.zeros([l, l])
    for i in range(l):
        for j in range(l):
            P[i, j] = get_trace(np.dot(np.dot(Un, train_x_k_list[i]), Un), np.dot(np.dot(Un, train_x_k_list[j]), Un.T))
            # P[j,i] = P[i,j]#对称矩阵
    return P

def get_q(train_x_k_list, ideal_kernel):
    '''
    input:
        train_x_k_list 训练集高斯矩阵
    output:
        train_y 标签
    '''
    l = len(train_x_k_list)
    n = len(train_x_k_list[0])
    # 计算Un Un = In - (1/n)ln*ln.T
    In = np.identity(n)
    ln = np.ones([n, 1])
    Un = In - (1 / n) * np.dot(ln, ln.T)
    # 计算Ki 理想核
    Ki = ideal_kernel
    # 计算a
    a = np.zeros([l, 1])
    for i in range(l):
        a[i, 0] = get_trace(np.dot(np.dot(Un, train_x_k_list[i]), Un), Ki)
    return a

def get_WW(t1 , t2):
    '''
    计算两个矩阵的余弦相似度
    '''
    fenzi = np.trace(np.dot(t1,t2))
    fenmu = ((np.trace(np.dot(t1 , t1)))*(np.trace(np.dot(t2 , t2))))**0.5
    return round(fenzi / fenmu , 4)

def getMB(np_array):
    """
    特征归一化算法 Moreau_Brota
    input:
        np_array 输入特征矩阵
    output:
        Broto_array归一化后的特征矩阵
    """
    Max = np_array.max(axis=0)  # 求平均值
    Min = np_array.min(axis=0)  # 求方差
    Broto_array = (np_array - Min) / (Max - Min)  # 归一化后的数组#广播#点成*#乘积dot
    return Broto_array  # 两种归一化方法

def getMB_double(train, test):
    """
    input:
        train
        test
    output:
        MB_ed train
        MB_ed test
    """
    n = len(train)
    train_test = np.vstack([train, test])  # 垂直拼接
    Max = train_test.max(axis=0)  # 求平均值
    Min = train_test.min(axis=0)  # 求方差
    Broto_array = (train_test - Min) / (Max - Min)  # 归一化后的数组#广播#点成*#乘积dot
    return Broto_array[0:n, :], Broto_array[n:, :]

def get_z(array):
    '''
    z-score 归一化算法
    x* = （ x- 均值 ） / 标准差
    '''
    AVE = array.mean(axis=0)
    STD = array.std(axis=0)
    return (array - AVE) / STD

def get_Med(array):
    '''
    对参数归一化
    M(i,j)/[M(j,j)**0.5  *  M(i,i)**0.5]
    '''
    l = len(array)
    re = np.zeros([l, l])
    for i in range(l):
        for j in range(l):
            re[i][j] = array[i][j] / ((array[i][i] ** 0.5) * (array[j][j] ** 0.5))
    return re

def get_mu(array):
    s = 0
    for i in range(len(array)):
        s = s + array[i] ** 2
    return array / (s ** 0.5)

def kernel_gussian(x, gamma):
    # 相似性矩阵计算
    n = len(x)
    kernel = np.zeros([n, n])
    for i in range(n):
        for j in range(i, n):
            kernel[i, j] = np.sum((x[i, :] - x[j, :]) ** 2)
            kernel[j, i] = kernel[i, j]

    return np.exp(-gamma * kernel)


def kernel_cosine(x, mu, sigma):
    # Calculates the link indicator kernel from a graph adjacency by cosine similiarity
    n = len(x)
    m = len(x[0])
    # Add Gaussian random noise matrix
    x = x + np.random.normal(mu, sigma, (n, m))
    kernel = np.zeros([n, n])
    for i in range(n):
        for j in range(i, n):
            kernel[i, j] = np.dot(x[i, :], x[j, :].T) / (np.linalg.norm(x[i, :]) * np.linalg.norm(x[j, :]))
            kernel[j, i] = kernel[i, j]
    return kernel

def kernel_corr(x, mu, sigma):
    # Calculates the link indicator kernel from a graph adjacency by pairwise linear correlation coefficient
    n = len(x)
    m = len(x[0])
    # Add Gaussian random noise matrix
    x = x + np.random.normal(mu, sigma, (n, m))
    return np.corrcoef(x)

def kernel_MI(x):
    n = len(x)
    m = len(x[0])
    kernel = np.zeros([n, n])
    for i in range(n):
        for j in range(i, n):
            kernel[i, j] = metrics.normalized_mutual_info_score(x[i, :], x[j, :])
            kernel[j, i] = kernel[i, j]
    return kernel

def kernel_normalized(k):
    # 理想核矩阵的归一化
    n = len(k)
    k = np.abs(k)
    index_nozeros = k.nonzero()
    min_value = min(k[index_nozeros])
    k[np.where(k == 0)] = min_value

    diag = np.resize(np.diagonal(k), [n, 1]) ** 0.5
    k_nor = k / (np.dot(diag, diag.T))
    return k_nor

def load_kernel_from_file(file_path):
    """"""
    return np.loadtxt(file_path)
    
def get_n_weight(k_train_list ,ideal_kernel ,lambd):
    '''
    input：
        k_train_list
    output：
        weight weight
    '''
    n = len(k_train_list)
    Wij = np.zeros([n,n])
    for i in range(n):#Wij
        for j in range(i,n):
            Wij[i][j] = get_WW(k_train_list[i],k_train_list[j])
            Wij[j][i] = Wij[i][j]
    D = Wij.sum(axis = 0)
    Dii = np.zeros([n,n])
    for i in range(n):
        Dii[i][i] = D[i]
    L = Dii - Wij
    L=abs(L)
    M = get_Med( get_P(k_train_list) ) #归一化
    P = M + lambd*L
    a = get_mu ( get_q(k_train_list , ideal_kernel) )
    q = -1*a
    G = -1 * np.identity(n)
    h = np.zeros([n,1])
    A = np.ones([1,n])
    b=matrix(1.)
    return get_CKA_Wi(P , q , G , h , A , b)

def get_train_label(y, cv_index, cv_i):
    n_drug = len(y)
    n_effect = len(y[0])
    y = y.reshape(n_drug * n_effect)
    y_label = y[cv_index == cv_i]
    y[np.where(cv_index == cv_i)] = 0
    y_train = y
    y_train = y_train.reshape([n_drug, n_effect])
    return y_train, y_label

def get_pre(y_pre, cv_index, cv_i):
    n_drug = len(y_pre)
    n_effect = len(y_pre[0])
    y_pre = y_pre.reshape(n_drug * n_effect)
    pre = y_pre[cv_index == cv_i]
    return pre

def get_auc_aupr(y_true, y_pre):
    auc_score = roc_auc_score(y_true, y_pre)
    aupr_score = average_precision_score(y_true, y_pre)
    return auc_score, aupr_score

def wknkn(y, Similar_1, Similar_2, knn, miu):
    n = len(y)
    m = len(y[0])

    y_d = np.zeros([n, m])
    y_t = np.zeros([m, n])

    index = np.argsort(Similar_1, 1)
    index = index[:, ::-1]
    index = index[:, :knn]

    for d in range(n):
        w_i = np.zeros([1, knn])
        for ii in range(knn):
            w_i[0, ii] = (miu ** ii) * Similar_1[d, index[d, ii]]
        # normalization term
        z_d = 1 / sum(Similar_1[d, index[d, :]])

        y_d[d, :] = z_d * (np.dot(w_i, y[index[d, :], :]))

    index = np.argsort(Similar_2, 1)
    index = index[:, ::-1]
    index = index[:, :knn]

    for t in range(m):
        w_i = np.zeros([1, knn])
        for ii in range(knn):
            w_i[0, ii] = (miu ** ii) * Similar_2[t, index[t, ii]]
        # normalization term
        z_t = 1 / sum(Similar_2[t, index[t, :]])

        y_t[t, :] = z_t * (np.dot(w_i, y.T[index[t, :], :]))

    y_dt = (y_d + y_t.T) / 2
    f_new = np.fmax(y, y_dt)
    return f_new

def getACC(y_true, y_pre):
    """
    y_true是n*1维度数组
    y_pre在模型计算后是1*n维数组
        这里y_pre.resize(len(y_pre),1)已经经过变换！！！
    """
    length = len(y_true)
    y_pre.resize(len(y_pre), 1)
    right = np.count_nonzero(y_true == y_pre)
    ACC = round(right / length, 4)
    return ACC

def my_confusion_matrix(y_true, y_pre):
    """
    y_true:numpy矩阵
    获得con矩阵
    TP  FP
    TN  FN
    """
    y_pre.resize(len(y_pre), 1)
    return confusion_matrix(y_true, y_pre)

def getMCC(con):
    TP = con[0, 0]
    FN = con[0, 1]
    FP = con[1, 0]
    TN = con[1, 1]
    MCC = (TP * TN - FP * FN) / ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5
    MCC = round(MCC, 4)
    return MCC

def getSN(con):
    TP = con[0, 0]
    FN = con[0, 1]
    SN = TP / (TP + FN)
    SN = round(SN * 100, 4)
    return SN

def getSP(con):
    FP = con[1, 0]
    TN = con[1, 1]
    SP = TN / (TN + FP)
    SP = round(SP * 100, 4)
    return SP

for s in itertools.product(m_threshold,epochs):

        association = pd.read_csv("similarity and feature/MD_A.csv", header=0, index_col=0).to_numpy()
        samples = get_all_samples(association)

        #fun_MS.txt,sem_DS.txt
        fun_sim = pd.read_csv("fun_MS.txt", header=None, index_col=None, sep='\t').to_numpy()
        sem_sim = pd.read_csv("sem_DS.txt", header=None, index_col=None, sep='\t').to_numpy()

        kf = KFold(n_splits=n_splits, shuffle=True)

        # MICROBE and disease features extraction from SVD
        SVD_mfeature = pd.read_csv('similarity and feature/SVD_microbe_feature.csv', header=None, index_col=None).values
        SVD_dfeature = pd.read_csv('similarity and feature/SVD_disease_feature.csv', header=None, index_col=None).values

        # k-fold cross-validation
        for train_index, val_index in kf.split(samples):
            fold += 1
            train_samples = samples[train_index, :]   # Training set of the current fold
            val_samples   = samples[val_index, :]     # Validation set of the current fold
            
            # -------------------------------------------------------------------
            # Step 1: Construct the association matrix for training only
            # We initialize a zero matrix with the same shape as the full association matrix,
            # and then fill in only the training associations. 
            # Importantly, we DO NOT use the validation/test associations here.
            # This ensures that similarity matrices are computed strictly from training data,
            # thereby avoiding any potential information leakage.
            # -------------------------------------------------------------------
            train_association = np.zeros_like(association)
            for i in train_samples:
                train_association[i[0], i[1]] = 1
            
            # -------------------------------------------------------------------
            # Step 2: Compute similarity matrices based on training data only
            # Cosine similarity and Gaussian kernel similarity are calculated using 
            # the training association matrix. Validation/test data are strictly excluded.
            # -------------------------------------------------------------------
            
            # Cosine similarity between microbes and diseases
            cos_MS, cos_DS = cos_sim(train_association)
            cos_MS = pd.DataFrame(cos_MS)  
            cos_DS = pd.DataFrame(cos_DS) 
            
            # Gaussian kernel similarity between microbes and diseases
            rm, rt = r_func(train_association)        # Parameters derived only from training data
            GaM, GaD = Gau_sim(train_association, rm, rt)
            GaM = pd.DataFrame(GaM)  
            GaD = pd.DataFrame(GaD)
        
            # -------------------------------------------------------------------
            # NOTE:
            # Unlike the previous version where validation associations were masked 
            # (set to zero) in the full matrix, this revised approach ensures 
            # that the validation set does not influence similarity computation 
            # in any form. This modification completely eliminates the possibility 
            # of structural information leakage.
            # -------------------------------------------------------------------


            n_microbe = len(train_association)  
            n_disease = len(train_association[0])     
            np.random.seed(2024)
            cv_index = np.random.randint(cv, size=n_microbe*n_disease)
            print("cv_index分布:", np.bincount(cv_index))
            
            k_train_list1 = [sem_sim,cos_DS,GaD]
            y_train, y_label = get_train_label(train_association, cv_index, cv_i)
            side_ideal_kernel = np.dot(y_train.T, y_train)
            side_k_nor = kernel_normalized(side_ideal_kernel)
            weights = get_n_weight(k_train_list1, side_k_nor, 0.8)
            k_s1 = np.zeros([n_disease, n_disease])
            for i in range(len(k_train_list)):
                k_s1 = k_s1 + weights[i] * k_train_list1[i]
            k_s1 = pd.DataFrame(k_s1)
            
            k_train_list2 = [fun_sim,cos_MS,GaM]
            y_train, y_label = get_train_label(train_association, cv_index, cv_i)
            side_ideal_kernel = np.dot(y_train, y_train.T)
            side_k_nor = kernel_normalized(side_ideal_kernel)
            weights = get_n_weight(k_train_list2, side_k_nor, 0.8)
            k_s2 = np.zeros([n_microbe, n_microbe])
            for i in range(len(k_train_list)):
                k_s2 = k_s2 + weights[i] * k_train_list2[i]
            k_s2 = pd.DataFrame(k_s2)
            
            # Microbe features extraction from GATE
            m_network = sim_thresholding( k_s2, s[0])
            m_adj, meta_features = generate_adj_and_feature(m_network, train_association)
            m_features = get_gae_feature(m_adj, meta_features, s[1], 1)

            # Disease features extraction from four-layer auto-encoder
            d_features = four_AE(k_s1,train_association)

            # get feature and label
            train_feature, train_label = generate_f1(D, train_samples, m_features, d_features, SVD_mfeature, SVD_dfeature)
            val_feature, val_label = generate_f1(D, val_samples, m_features, d_features, SVD_mfeature, SVD_dfeature)

            # modifyed CascadeForest
            model = CascadeForestClassifier(random_state=2024, verbose=1, n_jobs=-1)
            estimators = [
            RandomForestClassifier(random_state=2024, n_jobs=-1, n_estimators=100),
            XGBClassifier(random_state=2024, n_jobs=-1, n_estimators=100),
            ExtraTreesClassifier(random_state=2024, n_jobs=-1, n_estimators=100),
            LGBMClassifier(random_state=2024, n_jobs=-1, n_estimators=100, verbose=1)
            ]
            model.set_estimator(estimators)
            predictor = SVC(random_state=2024, probability=True, kernel='poly')
            model.set_predictor(predictor)
            model.fit(train_feature, train_label)
            test_N = val_samples.shape[0]
            y_score = np.zeros(test_N)
            y_score = model.predict(val_feature)[:, 0]

            # calculate metrics
            fpr, tpr, thresholds = roc_curve(val_label, y_score)
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)

            result += get_metrics(val_label, y_score)
            print('[aupr, auc, f1_score, accuracy, recall, specificity, precision]',
                  get_metrics(val_label, y_score))

        print("==================================================")
        print(result / n_splits)

        # plot ROC curve
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='g', label='diagonal', alpha=.8)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, color='r', label=r'CFAESVD ROC (area=%0.3f)' % mean_auc, lw=2, alpha=.8)
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='gray', alpha=.2)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC')
        plt.legend(loc='lower right')
        plt.show()
