import torch
from medscan.model import build_model

def test_resnet_output_shape():
    m=build_model(3,"resnet50",False).eval()
    with torch.no_grad(): y=m(torch.randn(1,3,224,224))
    assert y.shape==(1,3)
