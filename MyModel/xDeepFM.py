import torch
import layer


class xDeepFM(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, mlp_dims, dropout, cross_layer_sizes, split_half=True):
        super(xDeepFM, self).__init__()
        self.linear = layer.FeaturesLinear(field_dims)
        self.embedding = layer.FeaturesEmbedding(field_dims, embed_dim)
        self.mlp = layer.MultiLayerPerceptron(len(field_dims)*embed_dim, mlp_dims, dropout)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.cin = layer.CompressedInteractionNetwork(len(field_dims), cross_layer_sizes, split_half)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        part1 = self.linear(x)
        part2 = self.cin(embed_x)
        part3 = self.mlp(embed_x.view(-1, self.embed_output_dim))
        x = part1 + part2 + part3
        return torch.sigmoid(x.squeeze(1))
