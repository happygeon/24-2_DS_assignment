import torch
import torch.nn as nn

"""
정말 간단하게 논문 구현만 해봅시다
논문에서 제시한대로 구현하면 됩니다!!
데이터셋은 Fashion MNIST를 사용하기 때문에, 이미지의 크기(=img_dim)는 28 * 28 = 784입니다.
z_dim은 latent vector z의 차원이고, default로 64로 설정했습니다.

Hint: nn.Sequential을 사용하면 간단하게 구현할 수 있습니다.

"""


class Original_Generator(nn.Module):

  def __init__(self, z_dim, img_dim):
      super().__init__()
      self.gen = nn.Sequential(
          nn.Linear(z_dim, 256),
          nn.LeakyReLU(0.2),
          nn.Dropout(),
          nn.Linear(256, 512),
          nn.LeakyReLU(0.2),
          nn.Dropout(),
          nn.Linear(512, 1024),
          nn.LeakyReLU(0.2),
          nn.Linear(1024, img_dim),
          nn.Tanh()
      )
  def forward(self, x):
    return self.gen(x)