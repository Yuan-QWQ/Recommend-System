import torch
import layer


class FactorizationMachineModel(torch.nn.Module):
    def __init__(self, field_dims, output_dim=16):
        super(FactorizationMachineModel, self).__init__()
        self.linear = layer.FeaturesLinear(field_dims)
        self.embedding = layer.FeaturesEmbedding(field_dims, output_dim)

    def forward(self, x):
        """
        在paper中，n应该为sum（field_nums）,然而我们这里的n为len（field_nums） why?
        作者回答：使用独热编码的矩阵乘法等效于使用标签编码嵌入查找
        设 m = 200, n1 = len（field_nums）=39, n2 = sum（field_nums）=13674
        paper中，使用了pd.get_dummies，则为 m * n2 * k
        在这里，我们先使用了embedding，先onehot再嵌入查找，其中嵌入字典的shape为 n2 * k
        即，嵌入字典即为V向量，每个特征（n2）可以在字典中找到对应Vi向量
        所以使用独热编码的矩阵乘法等效于使用标签编码嵌入查找
        求和时因为只有n1个向量的x值为1，其余值为0，而这n1个向量即为embedding后的 n1 * k个向量
        所以求和时省略x的值大小，直接对n1个向量求和即可
        """
        v = self.embedding(x)
        part1 = torch.sum(v, dim=1) ** 2
        part2 = torch.sum(v ** 2, dim=1)
        inter = torch.sum(part1 - part2, dim=1) * 0.5
        return torch.sigmoid(self.linear(x).squeeze(1) + inter)

