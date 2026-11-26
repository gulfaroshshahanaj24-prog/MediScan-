import torch
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,roc_auc_score

@torch.no_grad()
def predict_all(model,loader,device):
    model.eval(); ys=[]; ps=[]
    for x,y in loader:
        ps.append(torch.softmax(model(x.to(device)),1).cpu().numpy()); ys.append(y.numpy())
    import numpy as np
    return np.concatenate(ys),np.concatenate(ps)

def metrics(y,p,classes):
    import numpy as np
    pred=p.argmax(1)
    r={"accuracy":float(accuracy_score(y,pred)),
       "per_class":classification_report(y,pred,labels=list(range(len(classes))),target_names=classes,output_dict=True,zero_division=0),
       "confusion_matrix":confusion_matrix(y,pred,labels=list(range(len(classes)))).tolist()}
    try:
        r["roc_auc"]=float(roc_auc_score(y,p[:,1])) if len(classes)==2 else float(roc_auc_score(y,p,multi_class="ovr",average="macro",labels=list(range(len(classes)))))
    except ValueError: r["roc_auc"]=None
    return r
