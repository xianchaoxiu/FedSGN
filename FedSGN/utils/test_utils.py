'''
Import necessary libraries
'''
import torch
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from utils.pca_utils import *
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor
'''
Data preprocessing 
'''

MIN_MAX = 1
STANDAR = 2
ROBUST = 3

# Choose the preprocessing methods:
def prep_data(dataX, prep_type=1):

    change_dataX = dataX.copy()
    featuresToScale = change_dataX.columns

    if prep_type == MIN_MAX:
        min_max = MinMaxScaler()
        change_dataX.loc[:,featuresToScale] = min_max.fit_transform(change_dataX[featuresToScale])
    elif prep_type == STANDAR:
        sX = StandardScaler(copy=True)
        change_dataX.loc[:,featuresToScale] = sX.fit_transform(change_dataX[featuresToScale])
    else:
        robScaler = RobustScaler()
        change_dataX.loc[:,featuresToScale] = robScaler.fit_transform(change_dataX[featuresToScale])

    return change_dataX


'''
 Define the score function for abnormal detection
'''
def anomalyScores(originalDF, reducedDF):
    loss = np.sum((np.array(originalDF) - np.array(reducedDF))**2, axis=1) 
    return loss


'''
Function test on all types of attacks from csv file
'''
def test_on_random_attack_samples_csv(V_k, thres_hold = 6, prep_type=MIN_MAX):
    dataX = pd.read_csv('abnormal_detection_data/test/abnormal_test_34_fea.csv', index_col=False)
    dataX = dataX.drop(['outcome', 'Unnamed: 0'], axis=1)
    dataX = prep_data(dataX, prep_type=STANDAR)
    data_transform = self_pca_transform_with_zero_mean(dataX, V_k)
    data_inverse = self_inverse_transform_with_zero_mean(data_transform, V_k)
    dataX = prep_data(dataX, prep_type=MIN_MAX)
    data_inverse = prep_data(data_inverse, prep_type=MIN_MAX)
    abnormal_score = anomalyScores(dataX, data_inverse)
    index = abnormal_score < thres_hold
    num_samples = dataX.shape[0]
    total_misclassified_samples = len(abnormal_score[index])
    return num_samples, total_misclassified_samples

'''
Test on another 10000 sample of normal data get data from csv file
'''
def test_on_normal_data_samples_csv(V_k, thres_hold=6):
    normal_data_10000 = pd.read_csv('abnormal_detection_data/test/kdd_10000_34_fea.csv', index_col=False)
    # Standardization 
    normal_data_10000 = prep_data(normal_data_10000, prep_type=STANDAR)
    # Transform data using Federated PCA components
    normal_data_pca = self_pca_transform_with_zero_mean(normal_data_10000, V_k)
    normal_data_inverse = self_inverse_transform_with_zero_mean(normal_data_pca, V_k)
    # Min max normalization before testing to keep balance between features
    normal_data_10000 = prep_data(normal_data_10000, prep_type=MIN_MAX)
    normal_data_inverse = prep_data(normal_data_inverse, prep_type=MIN_MAX)
    # Get anomalous score
    abnormal_score = anomalyScores(normal_data_10000, normal_data_inverse)
    # Get the misclassified samples
    index = abnormal_score > thres_hold
    total_normal_samples = normal_data_10000.shape[0]
    return total_normal_samples, len(abnormal_score[index])
    
'''
Test on both normal and abnormal data and calculate the F-Score
'''
def kdd_test(V_k, thres_hold):
    # Get total samples and misclassified samples on normal data
    normal_total_samples, normal_mis_samples = test_on_normal_data_samples_csv(V_k, thres_hold=thres_hold)
    FN = normal_mis_samples
    TN = normal_total_samples - normal_mis_samples
    # Get total samples and misclassified samples on abnormal data
    abnormal_total_samples, abnormal_mis_samples = test_on_random_attack_samples_csv(V_k, thres_hold=thres_hold)
    FP = abnormal_mis_samples
    TP = abnormal_total_samples - abnormal_mis_samples
    # Estimate performance based on the F-Score
    precision_score = TP/(FP + TP)
    recall_score = TP/(FN + TP)
    accuracy_score = (TP + TN)/ (TP + FN + TN + FP)
    f1_score = 2*precision_score*recall_score/(precision_score + recall_score)
    print(f"Precision: {precision_score * 100.0}")
    print(f"Recall: {recall_score * 100.0}")
    print(f"Accuracy score: {accuracy_score * 100.0}")
    print(f"F1 score: {f1_score * 100.0}")
    return accuracy_score*100.0

# Define the score function for abnormal detection which scale abnormal score in range [0:1]
def anomalyScores_0_1(originalDF, reducedDF):
    loss = np.sum((np.array(originalDF) - np.array(reducedDF))**2, axis=1)
    loss = (loss - np.min(loss)) / (np.max(loss) - np.min(loss))  # normalize
    
    return loss

# Calculate F-Score 
def results_analysis(df_gt_score, threshold, log=0):
  df_gt_pred = pd.DataFrame()
  df_gt_pred['ground_true'] = df_gt_score['ground_true']
  index = df_gt_score['anomalyScore'] > threshold
  df_gt_pred['prediction'] = index.astype(int)
  print(df_gt_pred)
  TN, FP, FN, TP = confusion_matrix(df_gt_pred['ground_true'], df_gt_pred['prediction']).ravel()
  precision_score = TP/(FP + TP)
  recall_score = TP/(FN + TP)
  accuracy_score = (TP + TN)/ (TP + FN + TN + FP)
  f1_score = 2*precision_score*recall_score/(precision_score + recall_score)
  fpr = FP / (FP+TN) # False positive rate (FPR); False Alarm = FP/N
  fng = FN / (TP+FN)

  if log:
    print(f"Precision: {np.round(precision_score * 100.0,4)}%")
    print(f"Recall: {np.round(recall_score * 100.0,4)}%")
    print(f"Accuracy score: {np.round(accuracy_score * 100.0,4)}%")
    print(f"F1 score: {np.round(f1_score * 100.0,4)}%")
    print(f"False alarm: {np.round(fpr * 100.0,4)}%")
    print(f"False Negative: {np.round(fng * 100.0,4)}%")

  return precision_score, recall_score, accuracy_score, f1_score, fpr, fng

# Test on NSL_KDD_Test+
def nsl_kdd_test(V_k):
    if isinstance(V_k, torch.Tensor):
        V_k_np = V_k.detach().cpu().numpy()
    else:
        V_k_np = V_k
    # Read data from csv files
    file_path_train = os.path.join(os.path.abspath(''), "abnormal_detection_data/train/nslkdd_train_normal.csv")
    file_path_test_normal = os.path.join(os.path.abspath(''), "abnormal_detection_data/test/nslkdd_test_normal.csv")
    file_path_test_abnormal = os.path.join(os.path.abspath(''), "abnormal_detection_data/test/nslkdd_test_abnormal.csv")
    df_train = pd.read_csv(file_path_train, index_col = 0)
    df_test_normal = pd.read_csv(file_path_test_normal, index_col = 0)
    df_test_abnormal = pd.read_csv(file_path_test_abnormal, index_col = 0)
    df_train = df_train.drop('outcome', axis=1)
    df_test_normal = df_test_normal.drop('outcome', axis=1)
    df_test_abnormal = df_test_abnormal.drop('outcome', axis=1)
    df_test = pd.concat([df_test_normal, df_test_abnormal])
    df_test.columns = df_test_abnormal.columns

    # Standardization over Testing
    scaler = StandardScaler()
    scaler.fit(df_train)
    df_test = pd.DataFrame(scaler.transform(df_test))
    df_test.columns = df_test_abnormal.columns
    
    # FedPCA: Obtain dataframe of groundTrue and anomalyScore using V_k
    df_test_transform = self_pca_transform_with_zero_mean(df_test, V_k_np)
    df_test_inverse = self_inverse_transform_with_zero_mean(df_test_transform, V_k_np)
    abnormal_score = anomalyScores_0_1(df_test, df_test_inverse)
    df_gt_score = pd.DataFrame(); df_gt_pred = pd.DataFrame()
    df_gt_score['ground_true'] = np.concatenate([np.zeros(len(df_test_normal)), np.ones(len(df_test_abnormal))])
    df_gt_score['anomalyScore'] = abnormal_score

    # Get results analysis
    precision_score, recall_score, accuracy_score, f1_score, fpr, fng =  results_analysis(df_gt_score, threshold=0.00025, log=1)

    return precision_score, recall_score, accuracy_score, f1_score, fpr, fng

# Test on UNSW NB15
def unsw_nb15_test(V_k):
    if isinstance(V_k, torch.Tensor):
        V_k_np = V_k.detach().cpu().numpy()
    else:
        V_k_np = V_k
    #Read data from csv files
    file_path_train = os.path.join(os.path.abspath(''), "abnormal_detection_data/train/unswnb15_train_normal.csv")
    file_path_test_normal = os.path.join(os.path.abspath(''), "abnormal_detection_data/test/unswnb15_test_normal_full.csv")
    file_path_test_abnormal = os.path.join(os.path.abspath(''), "abnormal_detection_data/test/unswnb15_test_abnormal.csv")
    df_train = pd.read_csv(file_path_train, index_col=0)
    df_test_normal = pd.read_csv(file_path_test_normal, index_col=0)
    df_test_abnormal = pd.read_csv(file_path_test_abnormal, index_col=0)    
    # Original UNSW-NB15 evaluation protocol in this project.
    df_test_normal = df_test_normal[:20000]
    df_test = pd.concat([df_test_normal, df_test_abnormal])
    df_test.columns = df_test_abnormal.columns  
    # 标准化
    scaler = StandardScaler()
    scaler.fit(df_train)
    df_test = pd.DataFrame(scaler.transform(df_test), columns=df_test_abnormal.columns) 
    # 映射到子空间 & 重构
    df_test_transform = self_pca_transform_with_zero_mean(df_test, V_k_np)
    df_test_inverse = self_inverse_transform_with_zero_mean(df_test_transform, V_k_np)  
    # 计算 anomaly score
    abnormal_score = anomalyScores_0_1(df_test, df_test_inverse)
    df_gt_score = pd.DataFrame()
    df_gt_score['ground_true'] = np.concatenate([np.zeros(len(df_test_normal)), np.ones(len(df_test_abnormal))])
    df_gt_score['anomalyScore'] = abnormal_score
    os.makedirs("results/FedSGN/scores", exist_ok=True)
    df_gt_score.to_csv("results/FedSGN/scores/unsw_nb15_lof_scores.csv", index=False)
    print("[Saved anomaly scores] results/FedSGN/scores/unsw_nb15_lof_scores.csv")
    # 评估指标
    # Tuned for the reproducible UNSW FedSGN model:
    # dim=10, k=15, seed=17, 20 rounds, 20,000 normal test samples.
    precision_score, recall_score, accuracy_score, f1_score, fpr, fng = results_analysis(
        df_gt_score, threshold=4.04862387792236e-06, log=1
    )
    return precision_score, recall_score, accuracy_score, f1_score, fpr, fng




def iot23_test_1(V_k):
    if isinstance(V_k, torch.Tensor):
        V_k_np = V_k.detach().cpu().numpy()
    else:
        V_k_np = V_k

    # === 读取 CSV ===
    file_path_train = os.path.join(os.path.abspath(''), "abnormal_detection_data/train/iot23_train_normal.csv")
    file_path_test_normal = os.path.join(os.path.abspath(''), "abnormal_detection_data/test/iot23_test_normal.csv")
    file_path_test_abnormal = os.path.join(os.path.abspath(''), "abnormal_detection_data/test/iot23_test_abnormal.csv")

    df_train = pd.read_csv(file_path_train, index_col=0)
    df_test_normal = pd.read_csv(file_path_test_normal, index_col=0)
    df_test_abnormal = pd.read_csv(file_path_test_abnormal, index_col=0)

    # === 构造测试集（正常 + 异常）===
    df_test = pd.concat([df_test_normal, df_test_abnormal])
    df_test.columns = df_test_abnormal.columns  # 对齐列顺序

    # === 标准化 ===
    scaler = StandardScaler()
    scaler.fit(df_train)
    df_test = pd.DataFrame(scaler.transform(df_test), columns=df_test_abnormal.columns)

    # === 投影到子空间 & 重构 ===
    df_test_transform = self_pca_transform_with_zero_mean(df_test, V_k_np)
    df_test_inverse = self_inverse_transform_with_zero_mean(df_test_transform, V_k_np)

    # === 计算异常分数 ===
    abnormal_score = anomalyScores_0_1(df_test, df_test_inverse)

    # === 构造带标签的数据框 ===
    df_gt_score = pd.DataFrame()
    df_gt_score['ground_true'] = np.concatenate([
        np.zeros(len(df_test_normal)),  # 正常为 0
        np.ones(len(df_test_abnormal)) # 异常为 1
    ])
    df_gt_score['anomalyScore'] = abnormal_score

    # === 保存得分 CSV ===
    os.makedirs("results/FedSGN/scores", exist_ok=True)
    df_gt_score.to_csv("results/FedSGN/scores/iot23_lof_scores.csv", index=False)
    print("[Saved anomaly scores] results/FedSGN/scores/iot23_lof_scores.csv")

    # === 评估指标 ===
    precision_score, recall_score, accuracy_score, f1_score, fpr, fng = results_analysis(df_gt_score, threshold=5.72e-1, log=1)

    return precision_score, recall_score, accuracy_score, f1_score, fpr, fng


def ton_test_1(V_k):
    if isinstance(V_k, torch.Tensor):
        V_k_np = V_k.detach().cpu().numpy()
    else:
        V_k_np = V_k

    # === 路径设定 ===
    file_path_train = os.path.join(os.path.abspath(''), "abnormal_detection_data/train/ton_train_normal_49.csv")
    file_path_test_normal = os.path.join(os.path.abspath(''), "abnormal_detection_data/test/ton_test_normal_49.csv")
    file_path_test_abnormal = os.path.join(os.path.abspath(''), "abnormal_detection_data/test/ton_test_abnormal_49.csv")
  
    # === 读取数据 ===
    df_train = pd.read_csv(file_path_train, index_col=0)
    df_test_normal = pd.read_csv(file_path_test_normal, index_col=0)
    df_test_abnormal = pd.read_csv(file_path_test_abnormal, index_col=0)

    # === 构造完整测试集 ===
    df_test = pd.concat([df_test_normal, df_test_abnormal])
    df_test.columns = df_test_abnormal.columns  # 对齐列顺序

    # === 标准化 ===
    scaler = StandardScaler()
    scaler.fit(df_train)
    df_test = pd.DataFrame(scaler.transform(df_test), columns=df_test_abnormal.columns)

    # === 映射到子空间并重构 ===
    df_test_transform = self_pca_transform_with_zero_mean(df_test, V_k_np)
    df_test_inverse = self_inverse_transform_with_zero_mean(df_test_transform, V_k_np)

    # === 计算异常得分 ===
    abnormal_score = anomalyScores_0_1(df_test, df_test_inverse)

    # === 标签 + 分数 ===
    df_gt_score = pd.DataFrame()
    df_gt_score['ground_true'] = np.concatenate([
        np.zeros(len(df_test_normal)), 
        np.ones(len(df_test_abnormal))
    ])
    df_gt_score['anomalyScore'] = abnormal_score

    # === 保存得分 CSV ===
    os.makedirs("results/FedSGN/scores", exist_ok=True)
    df_gt_score.to_csv("results/FedSGN/scores/ton_lof_scores.csv", index=False)
    print("[Saved anomaly scores] results/FedSGN/scores/ton_lof_scores.csv")

    # === 返回评估指标 ===
    precision_score, recall_score, accuracy_score, f1_score, fpr, fng = results_analysis(df_gt_score, threshold=4.701067379893356e-08, log=1)

    return precision_score, recall_score, accuracy_score, f1_score, fpr, fng









# Test on IoT23
def iot23_test(V_k):
  # Read data from csv files
  file_path_train = os.path.join(os.path.abspath(''), "abnormal_detection_data/train/iot23_train_normal.csv")
  file_path_test_normal = os.path.join(os.path.abspath(''), "abnormal_detection_data/test/iot23_test_normal.csv")
  file_path_test_abnormal = os.path.join(os.path.abspath(''), "abnormal_detection_data/test/iot23_test_abnormal.csv")
  df_train = pd.read_csv(file_path_train, index_col = 0)
  df_test_normal = pd.read_csv(file_path_test_normal, index_col = 0)
  df_test_abnormal = pd.read_csv(file_path_test_abnormal, index_col = 0)
#   df_train = df_train.drop("Unnamed: 0", axis=1)
#   df_test_normal = df_test_normal.drop("Unnamed: 0", axis=1)
#   df_test_abnormal = df_test_abnormal.drop("Unnamed: 0", axis=1)
  # df_test_normal = df_test_normal[:5000]
  df_test = pd.concat([df_test_normal, df_test_abnormal])
  df_test.columns = df_test_abnormal.columns

  # Standardization over Testing
  scaler = StandardScaler()
  scaler.fit(df_train)
  df_test = pd.DataFrame(scaler.transform(df_test))
  df_test.columns = df_test_abnormal.columns
  
  # FedPCA: Obtain dataframe of groundTrue and anomalyScore using V_k
  df_test_transform = self_pca_transform_with_zero_mean(df_test, V_k)
  df_test_inverse = self_inverse_transform_with_zero_mean(df_test_transform, V_k)
  abnormal_score = anomalyScores_0_1(df_test, df_test_inverse)
  df_gt_score = pd.DataFrame(); df_gt_pred = pd.DataFrame()
  df_gt_score['ground_true'] = np.concatenate([np.zeros(len(df_test_normal)), np.ones(len(df_test_abnormal))])
  df_gt_score['anomalyScore'] = abnormal_score
   
  # Get results analysis
  precision_score, recall_score, accuracy_score, f1_score, fpr, fng =  results_analysis(df_gt_score, threshold=1e-1000, log=1)
  print(df_gt_score)
  return precision_score, recall_score, accuracy_score, f1_score, fpr, fng


# Test on ToN IoT
def ton_test(V_k):
  # Read data from csv files
  file_path_train = os.path.join(os.path.abspath(''), "abnormal_detection_data/train/ton_train_normal_49.csv")
  file_path_test_normal = os.path.join(os.path.abspath(''), "abnormal_detection_data/test/ton_test_normal_49.csv")
  file_path_test_abnormal = os.path.join(os.path.abspath(''), "abnormal_detection_data/test/ton_test_abnormal_49.csv")
  df_train = pd.read_csv(file_path_train, index_col = 0)
  df_test_normal = pd.read_csv(file_path_test_normal, index_col = 0)
  df_test_abnormal = pd.read_csv(file_path_test_abnormal, index_col = 0)
#   df_train = df_train.drop("Unnamed: 0", axis=1)
#   df_test_normal = df_test_normal.drop("Unnamed: 0", axis=1)
#   df_test_abnormal = df_test_abnormal.drop("Unnamed: 0", axis=1)
  print(df_test_abnormal.shape)
  df_test_normal = df_test_normal[:10000]
  
  df_test = pd.concat([df_test_normal, df_test_abnormal])
  df_test.columns = df_test_abnormal.columns

  # Standardization over Testing
  scaler = StandardScaler()
  scaler.fit(df_train)
  df_test = pd.DataFrame(scaler.transform(df_test))
  df_test.columns = df_test_abnormal.columns
  
  # FedPCA: Obtain dataframe of groundTrue and anomalyScore using V_k
  df_test_transform = self_pca_transform_with_zero_mean(df_test, V_k)
  df_test_inverse = self_inverse_transform_with_zero_mean(df_test_transform, V_k)
  abnormal_score = anomalyScores_0_1(df_test, df_test_inverse)
  df_gt_score = pd.DataFrame(); df_gt_pred = pd.DataFrame()
  df_gt_score['ground_true'] = np.concatenate([np.zeros(len(df_test_normal)), np.ones(len(df_test_abnormal))])
  df_gt_score['anomalyScore'] = abnormal_score

  # Get results analysis
  precision_score, recall_score, accuracy_score, f1_score, fpr, fng =  results_analysis(df_gt_score, threshold=5e-6, log=1)
  
  return precision_score, recall_score, accuracy_score, f1_score, fpr, fng












def compute_unsup_threshold(scores, method="percentile", clip_p=None, q=40):
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return 0.0
    if method == "percentile":
        
        return float(np.percentile(s, q))





def synthetic_test(V_k):
    """
    在合成数据集上评估 FedSGN 的降维+异常检测效果。
    - 训练：synthetic_train_normal.csv
    - 测试：synthetic_test_normal.csv + synthetic_test_abnormal.csv
    """
    if isinstance(V_k, torch.Tensor):
        V_k_np = V_k.detach().cpu().numpy()
    else:
        V_k_np = V_k

    # === 数据路径 ===
    file_path_train        = os.path.join(os.path.abspath(''), "abnormal_detection_data/train/synthetic_train_normal.csv")
    file_path_test_normal  = os.path.join(os.path.abspath(''), "abnormal_detection_data/test/synthetic_test_normal.csv")
    file_path_test_abnormal = os.path.join(os.path.abspath(''), "abnormal_detection_data/test/synthetic_test_abnormal.csv")

    # === 读取数据（不要丢掉特征列） ===
    df_train        = pd.read_csv(file_path_train)
    df_test_normal  = pd.read_csv(file_path_test_normal)
    df_test_abnormal = pd.read_csv(file_path_test_abnormal)

    # （可选）只取一部分正常样本，防止过大
    if len(df_test_normal) > 20000:
        df_test_normal = df_test_normal.iloc[:20000]

    # 合并
    df_test = pd.concat([df_test_normal, df_test_abnormal], ignore_index=True)

    # === 标准化（按训练集分布）===
    scaler = StandardScaler()
    scaler.fit(df_train)
    df_test = pd.DataFrame(scaler.transform(df_test), columns=df_test.columns)

    # === PCA 投影 & 重构 ===
    df_test_transform = self_pca_transform_with_zero_mean(df_test, V_k_np)
    df_test_inverse   = self_inverse_transform_with_zero_mean(df_test_transform, V_k_np)

    # === 计算 anomaly score ===
    abnormal_score = anomalyScores_0_1(df_test, df_test_inverse)
    df_gt_score = pd.DataFrame({
        "ground_true": np.concatenate([
            np.zeros(len(df_test_normal)), 
            np.ones(len(df_test_abnormal))
        ]),
        "anomalyScore": abnormal_score
    })

    # 保存结果
    os.makedirs("results/FedSGN/scores", exist_ok=True)
    df_gt_score.to_csv("results/FedSGN/scores/synthetic_lof_scores.csv", index=False)
    print("[Saved anomaly scores] results/FedSGN/scores/synthetic_lof_scores.csv")

    # === 评估指标 ===
    precision_score, recall_score, accuracy_score, f1_score, fpr, fng = results_analysis(
        df_gt_score, threshold=0.3, log=1
    )
    return precision_score, recall_score, accuracy_score, f1_score, fpr, fng


 
