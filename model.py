import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest

class ResNet(nn.Module):
    def __init__(self, in_channels = 3 ,out_channels = 32):
        super().__init__()
        num_groups = 4
        if in_channels < 4:
            num_groups = in_channels
        self.num_channels = out_channels
        self.in_channels = in_channels
        self.network = nn.Sequential(
            nn.GroupNorm(num_groups,in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups,out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.network(x)
        return torch.add(out,self.residual_layer(x))

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = ResNet(1,8)
        self.q = nn.Linear(72,1)
        self.pi = nn.Linear(72,9)
    
    def forward(self, z):
        x = self.net(z)           
        x = x.view(z.size(0),-1)
        q = self.q(x)
        pi = self.pi(x)              
        return pi,q                       


# Unit test to verify the generator and discriminator
class TestModel(unittest.TestCase):
    def test_generator_output_shape(self):
        batch_size = 8
        gen = Model()

        # Generate random latent vectors
        z = torch.randn(batch_size, 1,3,3)

        # Generate images
        output,pi = gen(z)

        # Check that output shape is (batch_size, 3, 64, 64)
        self.assertEqual(output.shape, (batch_size, 1))


if __name__ == '__main__':
    # Run the unit tests
    unittest.main(argv=[''], exit=False)
