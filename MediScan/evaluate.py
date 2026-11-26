import argparse,json
from pathlib import Path
import torch
from medscan.data import loaders
from medscan.model import build_model
from medscan.evaluation import predict_all,metrics

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data-dir",default="data"); p.add_argument("--checkpoint",default="checkpoints/best_model.pt"); p.add_argument("--batch-size",type=int,default=16); a=p.parse_args()
    dev="cuda" if torch.cuda.is_available() else "cpu"; _,_,te,folder_classes=loaders(a.data_dir,a.batch_size)
    ck=torch.load(a.checkpoint,map_location=dev); classes=ck["class_names"]
    if classes!=folder_classes: raise ValueError("Dataset class order differs from checkpoint")
    m=build_model(len(classes),ck["architecture"],False).to(dev); m.load_state_dict(ck["model_state_dict"])
    y,pred=predict_all(m,te,dev); result=metrics(y,pred,classes)
    print(json.dumps(result,indent=2)); Path("reports").mkdir(exist_ok=True); Path("reports/test_metrics.json").write_text(json.dumps(result,indent=2))
if __name__=="__main__": main()
