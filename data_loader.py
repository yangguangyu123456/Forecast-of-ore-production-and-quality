import pandas as pd

def load_data(file_path):
    """
    加载数据集
    :param file_path: str, 数据集文件路径
    :return: DataFrame, 加载的数据集
    """
    return pd.read_csv(file_path)