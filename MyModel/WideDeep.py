import layer
import torch

"""
后续需要改进的点
1. 优化器的选择  wide的参数用ftrl+l1正则化优化 deep参数用sgd优化
2. wide 部分的特征嵌入需要进行特征处理
"""


class WideDeepModel(torch.nn.Module):
    def __init__(self, field_dims, output_dim, fc_dims, drop_out):
        super(WideDeepModel, self).__init__()
        self.linear = layer.FeaturesLinear(field_dims)
        self.embedding = layer.FeaturesEmbedding(field_dims, output_dim)
        self.mlp = layer.MultiLayerPerceptron(len(field_dims)*output_dim, fc_dims, drop_out)

    def forward(self, x):
        y = self.linear(x) + self.mlp(self.embedding(x).view(self.embedding(x).shape[0], -1))
        return torch.sigmoid(y).squeeze(1)
