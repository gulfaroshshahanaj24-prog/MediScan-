import argparse
from pathlib import Path
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from medscan.data import loaders
from medscan.model import build_model,freeze_backbone,unfreeze_all

def epoch(model,dl,loss_fn,device,opt=None):
    train=opt is not None; model.train(train); loss_sum=correct=count=0
    for x,y in dl:
        x,y=x.to(device),y.to(device)
        with torch.set_grad_enabled(train):
            out=model(x); loss=loss_fn(out,y)
            if train: opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        loss_sum+=loss.item()*len(y); correct+=(out.argmax(1)==y).sum().item(); count+=len(y)
    return loss_sum/max(count,1),correct/max(count,1)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-dir",default="data"); p.add_argument("--epochs",type=int,default=10)
    p.add_argument("--batch-size",type=int,default=16); p.add_argument("--image-size",type=int,default=224)
    p.add_argument("--learning-rate",type=float,default=1e-4); p.add_argument("--model",choices=["resnet50","efficientnet_b0"],default="resnet50")
    p.add_argument("--pretrained",choices=["true","false"],default="true"); p.add_argument("--freeze-epochs",type=int,default=2)
    p.add_argument("--output-dir",default="checkpoints"); a=p.parse_args()
    dev="cuda" if torch.cuda.is_available() else "cpu"
    tr,va,_,classes=loaders(a.data_dir,a.batch_size,a.image_size)
    m=build_model(len(classes),a.model,a.pretrained=="true").to(dev); freeze_backbone(m,a.model)
    opt=Adam(filter(lambda x:x.requires_grad,m.parameters()),lr=a.learning_rate)
    sched=ReduceLROnPlateau(opt,mode="min",patience=2,factor=.5); loss_fn=nn.CrossEntropyLoss()
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); best=-1
    for e in range(1,a.epochs+1):
        if e==a.freeze_epochs+1:
            unfreeze_all(m); opt=Adam(m.parameters(),lr=a.learning_rate*.1); sched=ReduceLROnPlateau(opt,mode="min",patience=2,factor=.5)
        tl,ta=epoch(m,tr,loss_fn,dev,opt); vl,vacc=epoch(m,va,loss_fn,dev); sched.step(vl)
        print(f"Epoch {e}/{a.epochs}: train loss={tl:.4f} acc={ta:.4f}; val loss={vl:.4f} acc={vacc:.4f}")
        if vacc>best:
            best=vacc; torch.save({"model_state_dict":m.state_dict(),"class_names":classes,"architecture":a.model,"image_size":a.image_size,"val_accuracy":vacc},out/"best_model.pt")
            print("Saved best checkpoint")
    print("Best validation accuracy:",best)
if __name__=="__main__": main()
