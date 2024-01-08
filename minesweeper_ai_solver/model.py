import torch
import torch.nn
import torch.nn.functional

from minesweeper import CellState


class MinesweeperSolver(torch.nn.Module):
    def __init__(self, dtype=None):
        super(MinesweeperSolver, self).__init__()

        self._model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=11, out_channels=64, kernel_size=3, padding='same', dtype=dtype),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same', dtype=dtype),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same', dtype=dtype),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same', dtype=dtype),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same', dtype=dtype),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, padding='same', dtype=dtype),
            torch.nn.Sigmoid()
        )

        #self._conv1 = torch.nn.Conv2d(in_channels=11, out_channels=64, kernel_size=3, padding='same', dtype=dtype)
        #self._conv2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same', dtype=dtype)
        #self._conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same', dtype=dtype)
        #self._conv4 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same', dtype=dtype)
        #self._conv5 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same', dtype=dtype)
        #self._conv6 = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, padding='same', dtype=dtype)

    def forward(self, x):
        x = self._model(x)
        # Last convolution returns the tensor with the size (num_batches, 1, input_shape[2:]) because it has only one
        # output chanel. Let's remove the dimension of size 1, because it will be more convenient to use the model
        # output in this form in the called code.
        output_shape = (x.size()[0],) + x.size()[2:]
        return x.view(output_shape)



