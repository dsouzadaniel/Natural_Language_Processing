# External Libraries
import torch
import torch.nn as nn


# Model Definition
class BiDAF(nn.Module):
    def __init__(self,
                 input_dim: int):
        super(BiDAF, self).__init__()
        # Model Properties
        self.input_dim = input_dim

        # Useful Constants
        self.CNN_FILTERS = 100
        self.CNN_WIDTH = 5

        # Model Layers
        self.cnn = nn.Conv1d(in_channels=self.input_dim,
                             out_channels=self.CNN_FILTERS,
                             kernel_size=self.CNN_WIDTH,
                             padding=self.CNN_WIDTH // 2,
                             stride=1
                             )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        output_tensor = self.cnn(input_tensor)
        max_output, _ = output_tensor.max(dim=2)
        max_output = max_output.squeeze()
        return max_output


bidaf = BiDAF(input_dim=1024)

input = torch.randn(1, 1024, 20)
print(input.shape)
output = bidaf(input)
print(output.shape)
