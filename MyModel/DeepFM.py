import layer
import torch


class DeepFactorizationMachineModel(torch.nn.Module):
    def __init__(self, field_dims, output_dim, fc_dims, drop_out):
        super(DeepFactorizationMachineModel, self).__init__()
        self.embedding = layer.FeaturesEmbedding(field_dims, output_dim)
        self.linear = layer.FeaturesLinear(field_dims)
        self.mlp = layer.MultiLayerPerceptron(len(field_dims)*output_dim, fc_dims, drop_out)

    def forward(self, x):
        v = self.embedding(x)
        part1 = torch.sum(v, dim=1) ** 2
        part2 = torch.sum(v ** 2, dim=1)
        inter = torch.sum(part1 - part2, dim=1) * 0.5
        y_fm = self.linear(x).squeeze(1) + inter
        y_mlp = self.mlp(v.view(v.shape[0], -1)).squeeze(1)
        return torch.sigmoid(y_fm + y_mlp)

