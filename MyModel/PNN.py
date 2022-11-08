import layer
import torch


class ProductBasedNeuralNetworkModel(torch.nn.Module):
    def __init__(self, field_dims, output_dim, fc_dims, drop_out, product='all'):
        super(ProductBasedNeuralNetworkModel, self).__init__()
        self.embedding = layer.FeaturesEmbedding(field_dims, output_dim)
        self.inner = layer.InnerProduct(field_dims)
        self.outer = layer.OuterProduct(field_dims, output_dim)
        if product == 'inner' or product == 'outer':
            mlp_input_dim = int(len(field_dims) + 0.5 * len(field_dims)*(len(field_dims)-1))
        else:
            mlp_input_dim = len(field_dims) + len(field_dims) * (len(field_dims) - 1)
        self.mlp = layer.MultiLayerPerceptron(mlp_input_dim*output_dim, fc_dims, drop_out)
        self.mode = product

    def forward(self, x):
        embeded = self.embedding(x)
        if self.mode == 'inner':
            part = self.inner(embeded)
        elif self.mode == 'outer':
            part = self.outer(embeded)
        else:
            part = torch.cat([self.inner(embeded), self.outer(embeded)], dim=1)
        mlp_input = torch.cat([embeded, part], dim=1)
        y = self.mlp(mlp_input.view(embeded.shape[0], -1))
        return torch.sigmoid(y).squeeze(1)


