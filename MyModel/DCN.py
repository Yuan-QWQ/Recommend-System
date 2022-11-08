import layer
import torch


class DeepCrossNetworkModel(torch.nn.Module):
    def __init__(self, field_dims, output_dim, cross_layers, fc_dims, drop_out):
        super(DeepCrossNetworkModel, self).__init__()
        self.embedding = layer.FeaturesEmbedding(field_dims, output_dim)
        self.mlp = layer.MultiLayerPerceptron(len(field_dims)*output_dim, fc_dims, drop_out, out_put=False)
        self.cn = layer.CrossNetwork(len(field_dims)*output_dim, cross_layers)
        self.linear = torch.nn.Linear(fc_dims[-1]+len(field_dims)*output_dim, 1)

    def forward(self, x):
        embeded = self.embedding(x).view(x.shape[0], -1)
        part1 = self.cn(embeded)
        part2 = self.mlp(embeded)
        x_stack = torch.cat([part1, part2], dim=1)
        y = self.linear(x_stack)
        return torch.sigmoid(y).squeeze(1)


