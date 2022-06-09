import torch


class GroupLinearLayer(torch.nn.Module):
    """From Recurrent Independent Mechanisms.
    https://github.com/anirudh9119/RIMs/blob/master/event_based/GroupLinearLayer.py
    """

    def __init__(self, input_dim, output_dim, num_blocks):
        super(GroupLinearLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = torch.nn.Parameter(
            torch.FloatTensor(num_blocks, self.input_dim, self.output_dim))
        # self.weight = torch.nn.Parameter(0.01 * torch.randn(num_blocks, din, dout))

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x = torch.bmm(x, self.weight)
        return x.permute(1, 0, 2)

    def reset_parameters(self):
        # torch.nn.init.xavier_uniform_(self.weight.data)
        torch.nn.init.normal_(self.weight.data, mean=0, std=np.sqrt(2.0 / (self.input_dim * 2)))