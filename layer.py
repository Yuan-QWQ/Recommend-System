import numpy
import numpy as np
import torch


# 等同于torch.nn.Linear
class FeaturesLinear(torch.nn.Module):
    def __init__(self, field_dims, output_dim=1):
        super(FeaturesLinear, self).__init__()
        """"
        假设sum(field_dims) = 10+20+10 = 40
        相当于构建了一个索引字典，索引为1到40，每个索引对应一个长度为output_dim=1的向量
        本质是先将输入onehot编码，然后执行onehot * weight
        即 m * n * sum(field_dims) 转化为 m * n * output_dim
        其中 sum(field_dims) -> output_dim 为全连接层
        m为数据量，n为特征个数，sum(field_dims)为特征总数
        """
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        # 将不可训练的tensor转换为可训练的parameter
        # why we should add an',' python list 或者tensor 代码后最后 加了一个逗号（，）就变成了元组
        self.bias = torch.nn.Parameter(torch.zeros(output_dim))
        self.offset = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        """
        x是经过自然数编码的值，因为filed_dims=[10, 20,10],那麽offsets=[0,10,30]，把[1, 3, 1] + [0, 10, 20] = [1, 13, 31]
        因为输入的x = [[1,3,1], [1,7,1], [2,10,2], [3,10,3], [4,11,2]]
        所以通过加上offsets之后 x变为了[1,13,31],[1,17,31],[2,20,32],[3,20,33],[4,21,32]]这一步是为了onehot
        """
        x = x + x.new_tensor(self.offset).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias


class FeaturesEmbedding(torch.nn.Module):
    def __init__(self, field_dims, output_dim):
        super(FeaturesEmbedding, self).__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), output_dim)

        self.offset = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        # 为什么层需要初始化？https://cloud.tencent.com/developer/article/1587082 默认为(0,1)正态分布
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        x = x + x.new_tensor(self.offset).unsqueeze(0)
        return self.embedding(x)


# class FactorizationMachine(torch.nn.Module):
#     def __init__(self):
#         super(FactorizationMachine, self).__init__()
#
#     def forward(self, x):
#         part1 = torch.sum(x, dim=1) ** 2
#         part2 = torch.sum(x**2, dim=1)
#         return torch.sum(part1 - part2, dim=1) * 0.5

class FieldAwareFactorization(torch.nn.Module):
    def __init__(self, field_dims, output_dim):
        super(FieldAwareFactorization, self).__init__()
        self.num_fields = len(field_dims)
        self.V = torch.nn.ModuleList(
            [FeaturesEmbedding(field_dims, output_dim) for _ in range(self.num_fields)]
        )

    def forward(self, x):
        xs = [self.V[i](x) for i in range(self.num_fields)]
        ix = list()
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                ix.append(xs[j][:, i] * xs[i][:, j])
        ix = torch.stack(ix, dim=1)
        return torch.sum(torch.sum(ix, dim=-1), dim=-1)


class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, input_dims, fc_dims, drop_out, out_put=True):
        super(MultiLayerPerceptron, self).__init__()
        layer = list()
        i = 0
        for dim in fc_dims:
            layer.append(torch.nn.Linear(input_dims, dim))
            torch.nn.init.kaiming_normal_(layer[i].weight.data)
            layer.append(torch.nn.BatchNorm1d(dim))
            layer.append(torch.nn.ReLU())
            layer.append(torch.nn.Dropout(p=drop_out))
            input_dims = dim
            i += 4
        if out_put:
            layer.append(torch.nn.Linear(input_dims, 1))
        self.mlp = torch.nn.Sequential(*layer)

    def forward(self, x):
        return self.mlp(x)


class InnerProduct(torch.nn.Module):
    def __init__(self, field_dims):
        super(InnerProduct, self).__init__()
        self.field_len = len(field_dims)

    def forward(self, x):
        inner = list()
        for i in range(self.field_len-1):
            for j in range(i+1, self.field_len):
                tmp = x[:, i] * x[:, j]
                inner.append(tmp)
        return torch.stack(inner, dim=1)


class OuterProduct(torch.nn.Module):
    def __init__(self, field_dims, output_dim):
        super(OuterProduct, self).__init__()
        self.field_len = len(field_dims)
        self.product_len = int(0.5 * len(field_dims) * (len(field_dims) - 1))
        self.V = torch.nn.ModuleList(
            [torch.nn.Linear(output_dim*output_dim, output_dim) for _ in range(self.product_len)]
        )

    def forward(self, x):
        outer = list()
        index = 0
        for i in range(self.field_len-1):
            for j in range(i + 1, self.field_len):
                tmp = torch.bmm(x[:, i].unsqueeze(-1), x[:, j].unsqueeze(-2)).reshape([x.shape[0], -1])
                outer.append(self.V[index](tmp))
                index += 1
        return torch.stack(outer, dim=1)


class CrossNetwork(torch.nn.Module):
    def __init__(self, fields, layer_nums):
        super(CrossNetwork, self).__init__()
        self.layers = layer_nums
        self.w = torch.nn.Parameter(torch.zeros(layer_nums, fields))
        self.b = torch.nn.Parameter(torch.zeros(layer_nums))
        torch.nn.init.xavier_uniform_(self.w)

    def forward(self, x):
        x_out = x
        for i in range(self.layers):
            x_out = x * self.w[i] * x_out + x_out + self.b[i]
        return x_out


class AttentionPooling(torch.nn.Module):
    def __init__(self, output_dim, attention_size, drop_out):
        super(AttentionPooling, self).__init__()
        self.linear = torch.nn.Linear(output_dim, attention_size)
        self.relu = torch.nn.ReLU()
        self.h = torch.nn.Parameter(torch.zeros(attention_size, 1))
        torch.nn.init.xavier_uniform_(self.h.data)
        self.softmax = torch.nn.Softmax(dim=1)
        self.drop_out = torch.nn.Dropout(p=drop_out)
        self.attention_size = attention_size

    def forward(self, x):
        y = self.relu(self.linear(x.view(-1, x.shape[-1])))
        y_ = (y * self.h.squeeze(-1)).view(-1, x.shape[1], self.attention_size)
        a = self.softmax(torch.sum(y_, dim=-1, keepdim=True))
        a = self.drop_out(a)
        return a


class CompressedInteractionNetwork(torch.nn.Module):
    def __init__(self, input_dim, cross_layer_sizes, split_half=True):
        super().__init__()
        self.num_layers = len(cross_layer_sizes)
        self.split_half = split_half
        self.conv_layers = torch.nn.ModuleList()
        prev_dim, fc_input_dim = input_dim, 0
        for i in range(self.num_layers):
            cross_layer_size = cross_layer_sizes[i]
            self.conv_layers.append(torch.nn.Conv1d(input_dim * prev_dim, cross_layer_size, 1,
                                                    stride=1, dilation=1, bias=True))
            if self.split_half and i != self.num_layers - 1:
                cross_layer_size //= 2
            prev_dim = cross_layer_size
            fc_input_dim += prev_dim
        self.fc = torch.nn.Linear(fc_input_dim, 1)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        xs = list()
        x0, h = x.unsqueeze(2), x
        for i in range(self.num_layers):
            x = x0 * h.unsqueeze(1)  # 内积
            batch_size, f0_dim, fin_dim, embed_dim = x.shape
            x = x.view(batch_size, f0_dim * fin_dim, embed_dim)
            x = torch.nn.functional.relu(self.conv_layers[i](x))
            if self.split_half and i != self.num_layers - 1:
                x, h = torch.split(x, x.shape[1] // 2, dim=1)
            else:
                h = x
            xs.append(x)
        return self.fc(torch.sum(torch.cat(xs, dim=1), 2))
