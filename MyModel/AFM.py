import layer
import torch


class AttentionalFactorizationMachinesModel(torch.nn.Module):
    def __init__(self, field_dims, output_dim, attention_size, drop_out):
        super(AttentionalFactorizationMachinesModel, self).__init__()
        self.linear = layer.FeaturesLinear(field_dims)
        self.embedding = layer.FeaturesEmbedding(field_dims, output_dim)
        self.inner = layer.InnerProduct(field_dims)
        t = int(len(field_dims) * (len(field_dims) - 1) * 0.5)
        self.attention = layer.AttentionPooling(t, output_dim, attention_size, drop_out)
        self.p = torch.nn.Parameter(torch.zeros(output_dim, 1))
        torch.nn.init.xavier_uniform_(self.p.data)

    def forward(self, x):
        part1 = self.linear(x)
        embeded = self.embedding(x)
        interaction = self.inner(embeded)
        a = self.attention(interaction)
        inter = torch.sum(a * interaction, dim=1)
        y = part1 + torch.mm(inter, self.p)
        return torch.sigmoid(y).squeeze(1)
