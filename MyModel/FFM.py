import torch
import layer


class FieldAwareFactorizationMachine(torch.nn.Module):
    def __init__(self, field_dims, output_dim=16):
        super(FieldAwareFactorizationMachine, self).__init__()
        self.linear = layer.FeaturesLinear(field_dims)
        self.ffm = layer.FieldAwareFactorization(field_dims, output_dim)

    def forward(self, x):
        y = self.linear(x).squeeze(1) + self.ffm(x)
        return torch.sigmoid(y)


