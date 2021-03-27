import torch
from randconv import randconv

def test_randconv():
  image = torch.randn(1, 3, 32, 32)
  assert (randconv(image, 10, False, 1.0) == image).to(torch.int).sum() == image.numel(), "Randconv standard failed"
  assert randconv(image, 4, True, 0.5) is not None, "Randconv mix failed"