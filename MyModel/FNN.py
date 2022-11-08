import layer
import torch


class FactorizationSupportedNeuralNetworkModel(torch.nn.Module):
    def __init__(self, field_dims, output_dim, fc_dims, drop_out):
        super(FactorizationSupportedNeuralNetworkModel, self).__init__()
        self.embedding = layer.FeaturesEmbedding(field_dims, output_dim)
        self.mlp = layer.MultiLayerPerceptron(len(field_dims)*output_dim, fc_dims, drop_out)

    def forward(self, x):
        y = self.mlp(self.embedding(x).view(self.embedding(x).shape[0], -1))
        return torch.sigmoid(y).squeeze(1)





