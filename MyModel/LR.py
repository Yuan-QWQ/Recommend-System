import torch
import layer


class LogisticRegressionModel(torch.nn.Module):
    def __init__(self, field_dims):
        super(LogisticRegressionModel, self).__init__()
        self.linear = layer.FeaturesLinear(field_dims)

    def forward(self, x):
        return torch.sigmoid(self.linear(x)).squeeze(1)



