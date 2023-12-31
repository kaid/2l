import numpy as np
from enum import Enum
from typing import Callable


def generate_weight(x: int, y: int) -> np.ndarray:
    return np.random.rand(x, y)


INPUT_NODE_NUM = 30  # 30个输入节点
HIDDEN_LAYER_1_NODE_NUM = 5  # 我们只有1个隐藏层, 5个节点
OUTPUT_NODE_NUM = 3  # 3个输出节点

WEIGHT1_SHAPE = (INPUT_NODE_NUM, HIDDEN_LAYER_1_NODE_NUM)  # 输入层到隐藏层的权重矩阵形状
WEIGHT2_SHAPE = (HIDDEN_LAYER_1_NODE_NUM, OUTPUT_NODE_NUM)  # 隐藏层到输出层的权重矩阵形状


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def softmax(x: np.ndarray) -> np.ndarray:
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean((y_true - y_pred) ** 2)


class Values(Enum):
    """
    对应到输出层最大值的索引
    """
    A = 0
    B = 1
    C = 2


class AbcModel:
    weight1: np.ndarray  # 输入到隐藏层的权重矩阵
    weight2: np.ndarray  # 隐藏层到输出层的权重矩阵
    activation: Callable[[np.ndarray], np.ndarray]  # 激活函数

    def __init__(
            self,
            activation=sigmoid,
            weight1=generate_weight(*WEIGHT1_SHAPE),
            weight2=generate_weight(*WEIGHT2_SHAPE)) -> None:
        self.weight1 = weight1
        self.weight2 = weight2
        self.activation = activation

    def __call__(self, *args, **kwargs) -> Values:
        return self.evaluate(*args, **kwargs)

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        # 隐藏层
        hidden = self.activation(np.dot(input_data, self.weight1))

        # 输出层
        return self.activation(np.dot(hidden, self.weight2))

    def evaluate(self, input_data: np.ndarray) -> Values:
        output = self.forward(input_data)
        return Values(np.argmax(output[0]))  # 返回最大值的索引, why?

    def back_propagation(self, input_data: np.ndarray, y_true: np.ndarray, learning_rate: float) -> float:
        # 隐藏层
        hidden = self.activation(np.dot(input_data, self.weight1))

        # 输出层
        output = self.activation(np.dot(hidden, self.weight2))

        # 输出层误差
        output_error = output - y_true

        # 隐藏层误差
        hidden_error = np.multiply(
            np.transpose(np.dot(self.weight2, np.transpose(output_error))),
            np.multiply(hidden, 1 - hidden))

        # 隐藏层误差梯度
        hidden_error_gradient = np.dot(np.transpose(input_data), hidden_error)

        # 输出层误差梯度
        output_error_gradient = np.dot(np.transpose(hidden), output_error)

        # 更新权重
        self.weight1 -= learning_rate * hidden_error_gradient
        self.weight2 -= learning_rate * output_error_gradient

        return mse(y_true, output)

    def train(
            self,
            input_dataset: list[np.ndarray],
            y_true: np.ndarray,
            learning_rate: float,
            epoch: int) -> (list[float], list[float]):
        loss_list = []
        accuracy_list = []
        input_dataset_size = len(input_dataset)

        for _ in range(epoch):
            epoch_loss_list = []
            for i, input_data in enumerate(input_dataset):
                label = y_true[i]
                output_data = self.forward(input_data)
                epoch_loss_list.append(mse(label, output_data))
                self.back_propagation(input_data, label, learning_rate)
            accuracy_list.append((1 - sum(epoch_loss_list) / input_dataset_size) * 100)
            loss_list.append(sum(epoch_loss_list) / input_dataset_size)

        return loss_list, accuracy_list
