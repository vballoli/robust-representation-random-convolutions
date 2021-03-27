import torch
from torchvision.models import resnet18

from randconv import consistency_loss, randconv

def test_cl():
    model = resnet18()
    image = torch.randn(1,3,64,64)

    out = model(image)
    rand_out_1 = model(randconv(image, 3, True, 0.5))
    rand_out_2 = model(randconv(image, 3, True, 0.5))
    rand_out_3 = model(randconv(image, 3, True, 0.5))
    cl = consistency_loss(0.1, rand_out_1, rand_out_2, rand_out_3)
    cl.backward()