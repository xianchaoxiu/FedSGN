import torch
import os
from FLAlgorithms.users.userADMM import UserADMM
from FLAlgorithms.users.userADMM2 import UserADMM2
from FLAlgorithms.servers.serverbase2 import Server2
from utils.store_utils import metrics_exp_store
from utils.test_utils import unsw_nb15_test, nsl_kdd_test, iot23_test, ton_test,synthetic_test
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler

''' Implementation for FedPCA Server'''

class AbnormalDetection(Server2):
    def __init__(self, algorithm, experiment, device, dataset, learning_rate, ro, num_glob_iters, local_epochs, clients, num_users, dim, time, exp_type):
        super().__init__(device, dataset, learning_rate, ro, num_glob_iters, local_epochs, num_users, dim, time)

        # Initialize data for all  users
        self.algorithm = algorithm
        self.local_epochs = local_epochs
        self.dataset = dataset 
        self.num_clients = clients
        # self.K = 0
        self.experiment = experiment
        self.experiment_type = exp_type

        if self.dataset == "Unsw":
            dataX = self.get_data_unsw_nb15()
            factor = dataX.shape[0]/self.num_clients # 56000 is total number of data points in UNSW NB15 dataset 
        elif self.dataset == "Iot23":
            dataX = self.get_data_Iot23()
            factor = dataX.shape[0]/self.num_clients # 29539 is total number of data points in IoT23 dataset 
        elif self.dataset == "Ton":
            dataX = self.get_data_Ton()
            factor = dataX.shape[0]/self.num_clients # 81872 is total number of data points in ToN IoT dataset
        elif self.dataset == "syn":
            dataX = self.get_data_synthetic_ab()
            factor = dataX.shape[0]/self.num_clients # 81872 is total number of data points in ToN IoT dataset
        else:
            dataX = self.get_data_snl_kdd()
            factor = dataX.shape[0]/self.num_clients # 67340 is total number of data points in NSL KDD dataset
        
        print(f"Total number of training samples: {dataX.shape[0]}")
        self.learning_rate = learning_rate
        self.user_fraction = num_users # percentage of total user involed in each global training round
        total_users = self.num_clients
        print("total users: ", total_users)
        for i in range(self.num_clients):            
            id = i
            train = self.get_client_data(dataX, factor=factor, i=i)
            train = torch.Tensor(train)
            if(i == 0):
                _, _, U = torch.svd(train) # This line of code is just to get the dimension of U matrix in the paper
                U = U[:, :dim]
                self.commonPCAz = torch.rand_like(U, dtype=torch.float) # the matrix U is randomized in the server
                # Give both methods the same feasible, orthonormal starting basis.
                if algorithm in {"FedPG", "FedPE"}:
                    self.commonPCAz, _ = torch.linalg.qr(self.commonPCAz, mode="reduced")

            user = UserADMM2(algorithm, device, id, train, self.commonPCAz, learning_rate, ro, local_epochs, dim)
            self.users.append(user)
            self.total_train_samples += user.train_samples
            
        print("Selected user in each Global Iteration / Total users:", int(num_users*total_users), " / " ,total_users)
        print("-------------------Finished creating FedPCA server-------------------")


    '''
    Get data from csv file
    '''
    def get_data(self, i):
        # Get data path
        directory = os.getcwd()
        print(f"directory: {directory}")
        data_path = os.path.join(directory, "abnormal_detection_data/train")
        print(data_path)
        file_name = f"client{i+1}_preprocessed.csv"
        client_path = os.path.join(data_path, file_name)
        print(client_path)

        # Read preprocessed data from csv file
        client_train = pd.read_csv(client_path)
        client_train = client_train.to_numpy()

        return client_train

    '''
    Get data from kdd dataset (.csv file)
    '''
    def get_data_kdd(self, i):
        # Get data path
        directory = os.getcwd()
        print(f"directory: {directory}")
        data_path = os.path.join(directory, "abnormal_detection_data/train")
        print(data_path)
        file_name = f"client{i+1}_kdd_std.csv"
        client_path = os.path.join(data_path, file_name)
        print(client_path)

        # Read preprocessed data from csv file
        client_train = pd.read_csv(client_path)
        client_train = client_train.to_numpy()

        return client_train
    
    '''
    Get data from nsl kdd dataset (.csv file)
    '''
    def get_data_snl_kdd(self):
        # Get data path
        directory = os.getcwd()
        print(f"directory: {directory}")
        data_path = os.path.join(directory, "abnormal_detection_data/train")
        print(data_path)
        file_name = f"nslkdd_train_normal.csv"
        client_path = os.path.join(data_path, file_name)
        print(client_path)

        # Read data from csv file and create non-i.i.d data for each client
        client_train = pd.read_csv(client_path)
        client_train = client_train.sort_values(by=['dst_bytes'])
        client_train = client_train.drop(['Unnamed: 0', 'outcome'], axis=1)
        print(client_train['dst_bytes'])
        print("Sorted!!!!!")

        return client_train

    '''
    Get data from unsw nb15 dataset (.csv file)
    '''
    def get_data_unsw_nb15(self):
        # Get data path
        directory = os.getcwd()
        print(f"directory: {directory}")
        data_path = os.path.join(directory, "abnormal_detection_data/train")
        print(data_path)
        file_name = f"unswnb15_train_normal.csv"
        client_path = os.path.join(data_path, file_name)
        print(client_path)

        # Read data from csv file and create non-i.i.d data for each client
        client_train = pd.read_csv(client_path)
        client_train = client_train.sort_values(by=['ct_srv_src'])
        client_train = client_train.drop(["Unnamed: 0"], axis=1)
        # print(client_train['ct_srv_src'])
        print("Created Non-iid Data!!!!!")

        return client_train


    

    '''
    Get data from Iot23 dataset (.csv file)
    '''
    def get_data_Iot23(self):
        # Get data path
        directory = os.getcwd()
        print(f"directory: {directory}")
        data_path = os.path.join(directory, "abnormal_detection_data/train")
        print(data_path)
        file_name = f"iot23_train_normal.csv"
        client_path = os.path.join(data_path, file_name)
        print(client_path)

        # Read data from csv file and create non-i.i.d data for each client
        client_train = pd.read_csv(client_path)
        client_train = client_train.sort_values(by=['duration'])
        client_train = client_train.drop(["Unnamed: 0"], axis=1)
        # print(client_train['duration'])
        print("Created Non-iid Data!!!!!")

        return client_train

    '''
    Get data from ToN dataset (.csv file)
    '''
    def get_data_Ton(self):
        # Get data path
        directory = os.getcwd()
        print(f"directory: {directory}")
        data_path = os.path.join(directory, "abnormal_detection_data/train")
        print(data_path)
        file_name = f"ton_train_normal_49.csv"
        client_path = os.path.join(data_path, file_name)
        print(client_path)

        # Read data from csv file and create non-i.i.d data for each client
        client_train = pd.read_csv(client_path)
        client_train = client_train.sort_values(by=['src_port'])
        client_train = client_train.drop(["Unnamed: 0"], axis=1)
        # print(client_train['dns_qtype'])
        print("Created Non-iid Data!!!!!")

        return client_train
    '''
    Preprocessing data step
    '''
    def prep_data(self, dataX):
        change_dataX = dataX.copy()
        featuresToScale = change_dataX.columns
        sX = StandardScaler(copy=True)
        change_dataX.loc[:,featuresToScale] = sX.fit_transform(change_dataX[featuresToScale])
        return change_dataX

    '''
    Divide data to clients
    '''
    def get_client_data(self, data, factor, i):
        # Read data frame for each client
        factor = int(factor)
        dataX = data[factor*i:factor*(i+1)].copy()
        # Preprocess data
        client_data = self.prep_data(dataX)
        client_data = client_data.to_numpy()
        return client_data
    

    '''
    Training model
    '''
    def train(self):
        current_loss = 0
        acc_score = 0
        losses_to_file = []
        acc_score_to_file = []
        acc_score_to_file.append(acc_score) # Initialize accuracy as zero
        self.selected_users = self.select_users(1000,1) # (1) Select all user in the network and distribute model to estimate performance in the first round (*)

        # Start estimating wall-clock time
        start_time = time.time()
        for glob_iter in range(self.num_glob_iters):
            if(self.experiment):
                self.experiment.set_epoch( glob_iter + 1)
            print("-------------Round number: ",glob_iter, " -------------")

            self.send_pca()

            # Evaluate model each interation
            current_loss = self.evaluate() # (*) The loss is estimated before training which requires (1)
            current_loss = current_loss.item()
            losses_to_file.append(current_loss)

            # Randomly choose a subset of users
            self.selected_users = self.select_users(glob_iter, self.user_fraction)

            # Train model in each user
            for user in self.selected_users:
                user.train(self.local_epochs)
                # print(f" selected user for training: {user.id}")
            self.aggregate_pca()

            # Evaluate the accuracy score
            Z = self.commonPCAz.detach().numpy()

            if self.dataset == "Unsw":
                precision_score, recall_score, accuracy_score, f1_score, fpr, fng = unsw_nb15_test(Z)
            elif self.dataset == "Iot23":
                precision_score, recall_score, accuracy_score, f1_score, fpr, fng = iot23_test(Z)
            elif self.dataset == "Ton":
                precision_score, recall_score, accuracy_score, f1_score, fpr, fng = ton_test(Z)
            elif self.dataset == "syn":
                precision_score, recall_score, accuracy_score, f1_score, fpr, fng = synthetic_test(Z)
            else:
                precision_score, recall_score, accuracy_score, f1_score, fpr, fng = nsl_kdd_test(Z)

            acc_score_to_file.append(accuracy_score)

        # End estimating wall-clock time
        end_time = time.time()

        # Extract common representation
        Z = self.commonPCAz.detach().numpy()
        
        # Extract losses to file
        losses_to_file = np.array(losses_to_file)

        # Extract accuracy score to file
        acc_score_to_file = np.array(acc_score_to_file)

        # Save common representation and losses to files
        # Get data path
        if self.algorithm == "FedPG":
            space = "Grassman"
        elif self.algorithm == "FedPE":
            space = "Euclidean"
        
        directory = os.getcwd()
        if self.dataset == "Unsw":
            data_path = os.path.join(directory, "results/UNSW")
            acc_path = os.path.join(data_path, "UNSW_acc")
            losses_path = os.path.join(data_path, "UNSW_losses")
            metrics_path = os.path.join(data_path, "UNSW_metrics_exp")
            model_dir = os.path.join(data_path, "UNSW_model")
        elif self.dataset == "Iot23":
            data_path = os.path.join(directory, "results/IOT23")
            acc_path = os.path.join(data_path, "IOT23_acc")
            losses_path = os.path.join(data_path, "IOT23_losses")
            metrics_path = os.path.join(data_path, "IOT23_metrics_exp")
            model_dir = os.path.join(data_path, "IOT23_model")
        elif self.dataset == "Ton":
            data_path = os.path.join(directory, "results/TON")
            acc_path = os.path.join(data_path, "TON_acc")
            losses_path = os.path.join(data_path, "TON_losses")
            metrics_path = os.path.join(data_path, "TON_metrics_exp")
            model_dir = os.path.join(data_path, "TON_model")
        elif self.dataset == "syn":
            data_path = os.path.join(directory, "results/syn")
            acc_path = os.path.join(data_path, "syn_acc")
            losses_path = os.path.join(data_path, "syn_losses")
            metrics_path = os.path.join(data_path, "syn_metrics_exp")
            model_dir = os.path.join(data_path, "syn_model")
        else:
            data_path = os.path.join(directory, "results/KDD")
            acc_path = os.path.join(data_path, "KDD_acc")
            losses_path = os.path.join(data_path, "KDD_losses")
            metrics_path = os.path.join(data_path, "KDD_metrics_exp")
            model_dir = os.path.join(data_path, "KDD_model")

        # Result directories are not included for every dataset in a fresh checkout.
        # Create them before saving so a first run can complete successfully.
        for output_dir in (acc_path, losses_path, metrics_path, model_dir):
            os.makedirs(output_dir, exist_ok=True)

        acc_file_name = f'{space}_acc_dim_{self.dim}_std_client_{self.num_clients}_iter_{self.num_glob_iters}_lr_{self.learning_rate}_sub_{self.user_fraction}_localEpochs_{self.local_epochs}'
        acc_file_path = os.path.join(acc_path, acc_file_name)
        losses_file_name = f"{space}_losses_dim_{self.dim}_std_client_{self.num_clients}_iter_{self.num_glob_iters}_lr_{self.learning_rate}_sub_{self.user_fraction}_localEpochs_{self.local_epochs}"
        losses_file_path = os.path.join(losses_path, losses_file_name)
        model_name = f'{space}_model_dim_{self.dim}_std_client_{self.num_clients}_iter_{self.num_glob_iters}_lr_{self.learning_rate}_sub_{self.user_fraction}_localEpochs_{self.local_epochs}'
        model_path = os.path.join(model_dir, model_name)

        # Store accuracy score to file
        np.save(acc_file_path, acc_score_to_file)
        np.save(losses_file_path, losses_to_file)
        np.save(model_path, Z)
        print(f"------------Final Test results------------")
        training_time = end_time - start_time
        print(f"training time: {training_time} seconds")

        if self.dataset == "Unsw":
            precision_score, recall_score, accuracy_score, f1_score, fpr, fng = unsw_nb15_test(Z)
        elif self.dataset == "Iot23":
            precision_score, recall_score, accuracy_score, f1_score, fpr, fng = iot23_test(Z)        
        elif self.dataset == "Ton":
            precision_score, recall_score, accuracy_score, f1_score, fpr, fng = ton_test(Z)
        elif self.dataset == "syn":
            precision_score, recall_score, accuracy_score, f1_score, fpr, fng = synthetic_test(Z)
        else:
            nsl_kdd_test(Z)

        # Store metrics experiment
        # Store metrics experiment
        metrics_file_name = f"{self.dataset}_{self.algorithm}_{self.experiment_type}.csv"
        metrics_exp_file_path = os.path.join(metrics_path, metrics_file_name)

        data_row = {
            'N_clients': self.num_clients,
            'Global_iter': self.num_glob_iters,
            'Local_iter': self.local_epochs,
            'dim': self.dim,
            'Loss': current_loss,
            'Acc': accuracy_score,
            'Precision': precision_score,
            'Recall': recall_score,
            'F1-score': f1_score,
            'FN': fng,
            'Time': training_time
        }

        metrics_exp_store(metrics_exp_file_path, data_row)
        print("Completed training!!!")
        print(f"------------------------------------------")



    def get_data_synthetic_ab(self):
        """
        读取人工合成数据集(正常样本)，可选混入一小部分异常样本以增强可分性。
        - 默认混入比例 10%（INJECT_RATIO），设为 0 则不混入。
        """
        # === 可调参数 ===
        INJECT_RATIO = 0
        SEED = 42

        # 路径
        directory = os.getcwd()
        train_normal_path = os.path.join(directory, "abnormal_detection_data/train/synthetic_train_normal.csv")
        test_abnormal_path = os.path.join(directory, "abnormal_detection_data/test/synthetic_test_abnormal.csv")

        # 读取正常训练集
        client_train = pd.read_csv(train_normal_path)
        for col in ["Unnamed: 0"]:
            if col in client_train.columns:
                client_train = client_train.drop(columns=[col])

        # 混入异常
        if INJECT_RATIO and INJECT_RATIO > 0:
            try:
                df_abn = pd.read_csv(test_abnormal_path)
            except Exception as e:
                print(f"[WARN] 读取异常样本失败：{e}，不混入异常。")
                df_abn = None

            if df_abn is not None:
                common_cols = sorted(list(set(client_train.columns).intersection(set(df_abn.columns))))
                if len(common_cols) == 0:
                    print("[WARN] 正常与异常列集合没有交集，放弃混入异常。")
                else:
                    client_train = client_train[common_cols].copy()
                    df_abn = df_abn[common_cols].copy()

                    n_add = max(1, int(INJECT_RATIO * len(client_train)))
                    n_add = min(n_add, len(df_abn))
                    abn_sample = df_abn.sample(n=n_add, random_state=SEED)

                    client_train = pd.concat([client_train, abn_sample], axis=0, ignore_index=True)
                    client_train = client_train.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
                    print(f"[FedSGN][Synthetic] 已混入异常样本: {n_add} / {len(client_train)} (ratio≈{INJECT_RATIO:.2%})")
            else:
                print("[FedSGN][Synthetic] 未混入异常样本。")
        else:
            print("[FedSGN][Synthetic] INJECT_RATIO=0，未混入异常样本。")

        print("Created Synthetic Non-iid Data!!!!!")
        return client_train
