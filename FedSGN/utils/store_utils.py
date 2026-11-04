import csv 
import os 
import time
# Setup contents
header = [
    'N_clients', 'Global_iter', 'Local_iter',
    'dim', 'k', 'learning_rate', 'ro',
    'Loss', 'Acc', 'Precision', 'Recall', 'F1-score', 'FN', 'Time'
]

def metrics_exp_store(file_path, data_row):
    """
    保存实验指标到 CSV，支持字段名自动对应。
    data_row: 必须是一个 dict，键为 header 中的字段名。
    """
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)

        # 如果文件不存在，写入表头
        if not file_exists:
            writer.writeheader()

        writer.writerow(data_row)


def print_log(message, fpath=None, stdout=True, print_time=False):
  if print_time:
    timestr = time.strftime('%Y-%m-%d %a %H:%M:%S')
    message = f'{timestr} | {message}'
  if stdout:
    print(message)
  if fpath is not None:
    with open(fpath, 'a') as f:
      print(message, file=f)