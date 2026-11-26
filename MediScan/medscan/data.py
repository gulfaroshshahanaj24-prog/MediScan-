from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

MEAN=(0.485,0.456,0.406)
STD=(0.229,0.224,0.225)

def loaders(root, batch=16, size=224, workers=2):
    root=Path(root)
    train_tf=transforms.Compose([transforms.Resize((size,size)), transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12), transforms.ColorJitter(brightness=.12,contrast=.12,saturation=.08),
        transforms.ToTensor(), transforms.Normalize(MEAN,STD)])
    eval_tf=transforms.Compose([transforms.Resize((size,size)),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    tr=datasets.ImageFolder(root/"train",transform=train_tf)
    va=datasets.ImageFolder(root/"val",transform=eval_tf)
    te=datasets.ImageFolder(root/"test",transform=eval_tf)
    if tr.classes != va.classes or tr.classes != te.classes: raise ValueError("Class folders must match in all splits")
    if not len(tr) or not len(va) or not len(te): raise ValueError("All splits must contain images")
    kw=dict(batch_size=batch,num_workers=workers)
    return DataLoader(tr,shuffle=True,**kw),DataLoader(va,shuffle=False,**kw),DataLoader(te,shuffle=False,**kw),tr.classes
