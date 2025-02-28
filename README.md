# 基于LSTM的矿石产量和质量预测

## 项目简介
本项目旨在使用长短期记忆网络（LSTM）对矿石的产量和质量进行预测。通过对历史数据的训练，模型可以对未来的矿石产量和质量进行准确的预测，以帮助矿山管理和决策。

## 文件结构
- `data/`：存放数据集
  - `train.csv`：训练数据集
  - `test.csv`：测试数据集
- `notebooks/`：存放Jupyter Notebook文件
  - `data_preprocessing.ipynb`：数据预处理
  - `model_training.ipynb`：模型训练
  - `model_evaluation.ipynb`：模型评估
- `src/`：存放源代码
  - `data_loader.py`：数据加载
  - `model.py`：模型定义
  - `train.py`：模型训练脚本
  - `evaluate.py`：模型评估脚本
- `requirements.txt`：项目依赖包
- `README.md`：项目说明文件

## 快速开始
1. 克隆仓库
    ```sh
    git clone https://github.com/yourusername/lstm-ore-prediction.git
    cd lstm-ore-prediction
    ```
2. 创建虚拟环境并安装依赖
    ```sh
    python -m venv venv
    source venv/bin/activate  # Windows系统使用 `venv\Scripts\activate`
    pip install -r requirements.txt
    ```
3. 运行数据预处理脚本
    ```sh
    jupyter notebook notebooks/data_preprocessing.ipynb
    ```
4. 训练模型
    ```sh
    jupyter notebook notebooks/model_training.ipynb
    ```
5. 评估模型
    ```sh
    jupyter notebook notebooks/model_evaluation.ipynb
    ```

## 贡献
欢迎提问和贡献代码！请阅读 `CONTRIBUTING.md` 了解详情。

## 许可证
本项目采用 MIT 许可证，详情请参阅 `LICENSE` 文件。