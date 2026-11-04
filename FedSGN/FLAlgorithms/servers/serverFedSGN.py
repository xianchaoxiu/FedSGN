import torch
import os
import time
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from FLAlgorithms.users.userFedSGN import UserFedSGN
from FLAlgorithms.servers.serverbase2 import Server2
from utils.store_utils import metrics_exp_store
from utils.optim_utils import sparse_newton_step
from utils.test_utils import (iot23_test_1,ton_test_1,unsw_nb15_test,synthetic_test,nsl_kdd_test)

class ServerFedSGN(Server2):


    def __init__(self,algorithm,experiment,device,dataset,learning_rate,
                 ro,num_glob_iters,local_epochs,clients,num_users,dim,time,exp_type,k):


        super().__init__(device,dataset,learning_rate,ro,
                         num_glob_iters,local_epochs,num_users,dim,time)

        self.algorithm = algorithm
        self.device=device
        self.experiment=experiment
        self.dataset=dataset
        self.num_clients=clients
        self.user_fraction=num_users
        self.local_epochs=local_epochs
        self.sparsity=k
        self.mu=ro
        self.learning_rate=learning_rate
        self.experiment_type=exp_type

#get datasets
        if self.dataset=="Unsw":
            dataX=self.get_data_unsw_nb15_ab()

        elif self.dataset=="Iot23":
            dataX=self.get_data_Iot23()

        elif self.dataset=="Ton":
            dataX=self.get_data_Ton_ab()

        elif self.dataset=="syn":
            dataX=self.get_data_synthetic_ab()

        else:
            dataX=self.get_data_snl_kdd()

        factor=dataX.shape[0]/clients
        print("Total samples:",dataX.shape[0])

#initialize Y
        self.d=dataX.shape[1]
        rand_matrix=torch.randn(self.d,self.dim,device=self.device)
        self.Y,_=torch.linalg.qr(rand_matrix)

#create clients
        self.users=[]
        self.total_train_samples = 0

        for i in range(self.num_clients):

            client_data=self.get_client_data(dataX,factor,i)

            X_tensor=torch.tensor(client_data,dtype=torch.float32).to(self.device)

            user=UserFedSGN(device=self.device,id=i,train_data=X_tensor,common_Y=self.Y.clone(),beta=None,local_epochs=self.local_epochs)

            self.users.append(user)

            self.total_train_samples += X_tensor.shape[0]

        print("Finished creating FedSGN server")

    def train(self):
        start_time=time.time()
        losses=[]
        comm_log=[]
        total_comm=0

        for glob_iter in range(self.num_glob_iters):
            print(f"\n[FedSGN] Round {glob_iter+1}/{self.num_glob_iters}")

#broadcast Y
            for user in self.users:
                user.set_global_Y(self.Y)

#record communication costs
            num_clients=len(self.users)
            param_MB=(self.Y.numel()*4/(1024**2))
            round_comm=(2*num_clients*param_MB)
            total_comm+=round_comm
            comm_log.append({"round":glob_iter+1,"clients":num_clients,"comm_MB":round_comm,"total_MB":total_comm})


            Z_list=[]
            beta_list=[]

            for user in self.users:
                X_i,Lambda_i,beta_i=user.get_model()
                Z_i=(X_i-Lambda_i/beta_i)
                Z_list.append(Z_i)
                beta_list.append(beta_i)

# Algorithm 1: solve the Y-subproblem first using X_i^k and
# Lambda_i^k.  The original implementation had these two blocks reversed.
            for _ in range(max(1, self.local_epochs)):
                self.Y, info=sparse_newton_step(
                    self.Y,
                    Z_list,
                    beta_list,
                    self.sparsity,
                    self.mu
                )

# Then update X_i^{k+1} with the newly computed Y^{k+1}.
            for user in self.users:
                user.set_global_Y(self.Y)
                user.update_primal()

#broadcast Y and update lamda
            for user in self.users:
            
                user.update_dual(
                    self.Y
                )


            selected_features = info["support"].numpy()

#evaluation
            acc,f1,prec,recall,fpr,fnr=\
                self.evaluate_on_selected_features(
                    selected_features
                )
            loss_dict=self.compute_global_loss()
            losses.append(
                loss_dict["total_loss"]
            )
            print(
                f"Loss:{loss_dict['total_loss']:.6f}"
            )
            print(
                f"Acc:{acc:.4f}, F1:{f1:.4f}"
            )

        training_time = time.time() - start_time
        print(f"Training time: {training_time:.2f}s")

#save result
        directory = os.getcwd()

        data_path = os.path.join(
            directory,
            "results/FedSGN"
        )

        acc_path = os.path.join(
            data_path,
            "acc"
        )

        loss_path = os.path.join(
            data_path,
            "loss"
        )

        model_path = os.path.join(
            data_path,
            "model"
        )

        metrics_path = os.path.join(
            data_path,
            "metrics"
        )

        comunication_path = os.path.join(
                    data_path,
                    "communication"
                )


        os.makedirs(
            acc_path,
            exist_ok=True
        )

        os.makedirs(
            loss_path,
            exist_ok=True
        )

        os.makedirs(
            model_path,
            exist_ok=True
        )

        os.makedirs(
            metrics_path,
            exist_ok=True
        )

        file_suffix = (
            f"dim_{self.dim}_"
            f"k_{self.sparsity}_"
            f"client_{self.num_clients}_"
            f"iter_{self.num_glob_iters}_"
            f"lr_{self.learning_rate}"
        )

        np.save(
            os.path.join(
                acc_path,
                f"acc_{file_suffix}.npy"
            ),
            np.array([acc])
        )

        np.save(
            os.path.join(
                loss_path,
                f"loss_{file_suffix}.npy"
            ),
            np.array(losses)
        )

        np.save(
            os.path.join(
                model_path,
                f"Y_{file_suffix}.npy"
            ),
            self.Y.detach().cpu().numpy()
        )

      

#communication cost
        comm_df = pd.DataFrame(
            comm_log
        )
        comm_df.to_csv(
            os.path.join(
                comunication_path,
                f"communication_{file_suffix}.csv"
            ),
            index=False
        )


#metrics restore
        metrics_file = os.path.join(
            metrics_path,
            f"{self.dataset}_FedSGN.csv"
        )

        data_row = {

            "N_clients":
                self.num_clients,

            "Global_iter":
                self.num_glob_iters,

            "Local_iter":
                self.local_epochs,

            "dim":
                self.dim,

            "k":
                self.sparsity,

            "learning_rate":
                self.learning_rate,

            "Loss":
                loss_dict["total_loss"],

            "Acc":
                acc,

            "Precision":
                prec,

            "Recall":
                recall,

            "F1-score":
                f1,

            "FN":
                fnr,

            "Time":
                training_time
        }

        metrics_exp_store(
            metrics_file,
            data_row
        )

# end
        print(
            "Completed training!!!"
        )

        return self.Y.detach().cpu()
   
#
    def evaluate_on_selected_features(
            self,
            topk_indices):

        if self.dataset == "Unsw":
            precision_score, recall_score, accuracy_score, f1_score, fpr, fnr = \
                unsw_nb15_test(self.Y)

        elif self.dataset == "Iot23":
            precision_score, recall_score, accuracy_score, f1_score, fpr, fnr = \
                iot23_test_1(self.Y)

        elif self.dataset == "Ton":
            precision_score, recall_score, accuracy_score, f1_score, fpr, fnr = \
                ton_test_1(self.Y)

        elif self.dataset == "syn":
            precision_score, recall_score, accuracy_score, f1_score, fpr, fnr = \
                synthetic_test(self.Y)

        else:

            precision_score, recall_score, accuracy_score, f1_score, fpr, fnr = \
                nsl_kdd_test(self.Y)

        print(
            f"[Evaluation] "
            f"Acc:{accuracy_score:.4f}, "
            f"F1:{f1_score:.4f}"
        )

        return (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            fpr,
            fnr
        )


    def compute_global_loss(self):

        projection_loss = 0.0
        dual_loss = 0.0
        consistency_loss = 0.0
        for user in self.users:
            X_i, Lambda_i, beta_i = user.get_model()
            C_i = user.C_i
            d = C_i.shape[0]
            I = torch.eye(d,device=self.device)
            gamma_i = user.gamma_i
            projection_loss += (0.5*torch.trace(X_i.T@(gamma_i*I-C_i)@X_i).item())
            dual_loss += (-torch.sum(Lambda_i*(X_i-self.Y)).item())
            consistency_loss += (0.5*beta_i*torch.norm(X_i-self.Y,p='fro')**2).item()

        d,p = self.Y.shape

        I_p = torch.eye(
            p,
            device=self.device
        )

        orthogonal_loss = (self.mu/4*torch.norm(self.Y.T@self.Y-I_p,p='fro')**2).item()

        total_loss = (projection_loss+dual_loss+consistency_loss+orthogonal_loss)

        return {

            "total_loss":
                total_loss,

            "proj":
                projection_loss,

            "dual":
                dual_loss,

            "consistency":
                consistency_loss,

            "orthogonal":
                orthogonal_loss

        }
    
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
        client_train = client_train.sort_values(by=['dst_bytes'])
        client_train = client_train.drop(["Unnamed: 0"], axis=1)
        # print(client_train['dns_qtype'])
        print("Created Non-iid Data!!!!!")

        return client_train

    def get_data_unsw_nb15_ab(self):

        # === 可调参数 ===
        INJECT_RATIO = 0          # Strict evaluation: never train on test anomalies.
        ABN_PATH = os.path.join(os.getcwd(), "abnormal_detection_data/test/unswnb15_test_abnormal.csv")
        SEED = 42

        # 读取正常训练集
        directory = os.getcwd()
        print(f"directory: {directory}")
        data_path = os.path.join(directory, "abnormal_detection_data/train")
        print(data_path)
        client_path = os.path.join(data_path, "unswnb15_train_normal.csv")
        print(client_path)

        client_train = pd.read_csv(client_path)
        # 去掉可能存在的索引列
        for col in ["Unnamed: 0"]:
            if col in client_train.columns:
                client_train = client_train.drop(columns=[col])

        # 排序保证和原实现一致（若列不存在则忽略）
        if "ct_srv_src" in client_train.columns:
            client_train = client_train.sort_values(by=["ct_srv_src"])

        # ====== 混入少量异常样本（可选）======
        if INJECT_RATIO and INJECT_RATIO > 0:
            try:
                df_abn = pd.read_csv(ABN_PATH, index_col=0)
            except Exception as e:
                print(f"[WARN] 读取异常样本失败：{e}，将不混入异常。")
                df_abn = None

            if df_abn is not None:
                # 对齐列（有些文件可能多/少列，这里做交并处理）
                # 目标：只保留在两边都存在的特征列
                common_cols = sorted(list(set(client_train.columns).intersection(set(df_abn.columns))))
                if len(common_cols) == 0:
                    print("[WARN] 正常与异常列集合没有交集，放弃混入异常。")
                else:
                    client_train = client_train[common_cols].copy()
                    df_abn = df_abn[common_cols].copy()

                    # 采样一定比例的异常样本（按正常样本数量的比例）
                    n_add = max(1, int(INJECT_RATIO * len(client_train)))
                    n_add = min(n_add, len(df_abn))  # 不超过可用异常样本数
                    abn_sample = df_abn.sample(n=n_add, random_state=SEED)

                    # 拼接 + 打乱
                    client_train = pd.concat([client_train, abn_sample], axis=0, ignore_index=True)
                    client_train = client_train.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
                    print(f"[FedSGN][UNSW] 已混入异常样本: {n_add} / {len(client_train)} (ratio≈{INJECT_RATIO:.2%})")
            else:
                print("[FedSGN][UNSW] 未混入异常样本。")
        else:
            print("[FedSGN][UNSW] INJECT_RATIO=0，未混入异常样本。")

        print("Created Non-iid Data!!!!!")
        return client_train

    def get_data_Ton_ab(self):

        INJECT_RATIO = 0
        ABN_PATH = os.path.join(os.getcwd(), "abnormal_detection_data/test/ton_test_abnormal_49.csv")
        SEED = 42

        # 读取正常训练集
        directory = os.getcwd()
        print(f"directory: {directory}")
        data_path = os.path.join(directory, "abnormal_detection_data/train")
        print(data_path)
        client_path = os.path.join(data_path, "ton_train_normal_49.csv")
        print(client_path)

        client_train = pd.read_csv(client_path)
        # 去掉可能存在的索引列
        for col in ["Unnamed: 0"]:
            if col in client_train.columns:
                client_train = client_train.drop(columns=[col])

        # 排序保证和原实现一致（若列不存在则忽略）
        if "dst_bytes" in client_train.columns:
            client_train = client_train.sort_values(by=["dst_bytes"])

        # ====== 混入少量异常样本（可选）======
        if INJECT_RATIO and INJECT_RATIO > 0:
            try:
                df_abn = pd.read_csv(ABN_PATH, index_col=0)
            except Exception as e:
                print(f"[WARN] 读取异常样本失败：{e}，将不混入异常。")
                df_abn = None

            if df_abn is not None:
                # 对齐列（有些文件可能多/少列，这里做交并处理）
                common_cols = sorted(list(set(client_train.columns).intersection(set(df_abn.columns))))
                if len(common_cols) == 0:
                    print("[WARN] 正常与异常列集合没有交集，放弃混入异常。")
                else:
                    client_train = client_train[common_cols].copy()
                    df_abn = df_abn[common_cols].copy()

                    # 采样一定比例的异常样本（按正常样本数量的比例）
                    n_add = max(1, int(INJECT_RATIO * len(client_train)))
                    n_add = min(n_add, len(df_abn))  # 不超过可用异常样本数
                    abn_sample = df_abn.sample(n=n_add, random_state=SEED)

                    # 拼接 + 打乱
                    client_train = pd.concat([client_train, abn_sample], axis=0, ignore_index=True)
                    client_train = client_train.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
                    print(f"[FedSGN][ToN] 已混入异常样本: {n_add} / {len(client_train)} (ratio≈{INJECT_RATIO:.2%})")
            else:
                print("[FedSGN][ToN] 未混入异常样本。")
        else:
            print("[FedSGN][ToN] INJECT_RATIO=0，未混入异常样本。")

        print("Created Non-iid Data!!!!!")
        return client_train

    def get_client_data(self, data, factor, i):
        # Read data frame for each client
        factor = int(factor)
        dataX = data[factor*i:factor*(i+1)].copy()
        # Preprocess data
        client_data = self.prep_data(dataX)
        client_data = client_data.to_numpy()
        return client_data


    def prep_data(self, dataX):
        change_dataX = dataX.copy()
        change_dataX = change_dataX.astype(float)
        featuresToScale = change_dataX.columns
        sX = StandardScaler(copy=True)
        change_dataX.loc[:,featuresToScale] = sX.fit_transform(change_dataX[featuresToScale])
        return change_dataX

    def get_data_synthetic_ab(self):

        # === 可调参数 ===
        INJECT_RATIO = 0.1
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
