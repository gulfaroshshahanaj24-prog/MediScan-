import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

class GradCAM:
    def __init__(self,model,layer):
        self.model=model; self.a=None; self.g=None
        self.h1=layer.register_forward_hook(lambda m,i,o:setattr(self,"a",o))
        self.h2=layer.register_full_backward_hook(lambda m,gi,go:setattr(self,"g",go[0]))
    def generate(self,x):
        self.model.zero_grad(set_to_none=True); logits=self.model(x)
        idx=int(logits.argmax(1).item()); logits[:,idx].sum().backward()
        w=self.g.mean((2,3),keepdim=True); cam=F.relu((w*self.a).sum(1,keepdim=True))
        cam=F.interpolate(cam,size=x.shape[-2:],mode="bilinear",align_corners=False)[0,0].detach().cpu().numpy()
        cam-=cam.min()
        if cam.max()>0: cam/=cam.max()
        return cam,logits.detach()
    def close(self): self.h1.remove(); self.h2.remove()

def overlay(image,cam,alpha=.4):
    import matplotlib.cm as cm
    im=image.convert("RGB").resize((cam.shape[1],cam.shape[0]))
    a=np.asarray(im).astype("float32")/255
    heat=cm.get_cmap("jet")(cam)[...,:3]
    return Image.fromarray((np.clip((1-alpha)*a+alpha*heat,0,1)*255).astype("uint8"))
