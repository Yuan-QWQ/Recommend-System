import layer
import torch


class NeuralFactorizationMachinesModel(torch.nn.Module):
    def __init__(self, field_dims, output_dim, fc_dims, drop_out):
        super(NeuralFactorizationMachinesModel, self).__init__()
        self.embedding = layer.FeaturesEmbedding(field_dims, output_dim)
        self.linear = layer.FeaturesLinear(field_dims)
        self.BI = layer.InnerProduct(field_dims)
        BI_output = 0.5*len(field_dims)*(len(field_dims)-1)*output_dim
        self.mlp = layer.MultiLayerPerceptron(int(BI_output), fc_dims, drop_out)

    def forward(self, x):
        # part1 = self.linear(x)
        part1 = 0
        embeded = self.embedding(x)
        BI_output = self.BI(embeded)
        part2 = self.mlp(BI_output.view(x.shape[0], -1))
        return torch.sigmoid(part1+part2).squeeze(1)

