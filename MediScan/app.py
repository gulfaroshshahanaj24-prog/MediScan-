import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path
from medscan.data import MEAN,STD
from medscan.model import build_model
from medscan.gradcam import GradCAM,overlay

st.set_page_config(page_title="MediScan",page_icon="🩺")
st.title("MediScan — Medical Image Classification")
st.warning("Educational demonstration only. Not clinically validated. Do not use for diagnosis or treatment.")
device="cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load():
    path=Path("checkpoints/best_model.pt")
    if not path.exists(): return None,None
    ck=torch.load(path,map_location=device); m=build_model(len(ck["class_names"]),ck["architecture"],False)
    m.load_state_dict(ck["model_state_dict"]); m.to(device).eval(); return m,ck

model,ck=load()
if model is None: st.error("Checkpoint missing. Train first using python train.py."); st.stop()
up=st.file_uploader("Upload an image",type=["jpg","jpeg","png","bmp"])
if up:
    image=Image.open(up).convert("RGB"); size=ck.get("image_size",224)
    tf=transforms.Compose([transforms.Resize((size,size)),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    x=tf(image).unsqueeze(0).to(device)
    layer=model.layer4[-1] if ck["architecture"]=="resnet50" else model.features[-1]
    gc=GradCAM(model,layer)
    try: cam,logits=gc.generate(x)
    finally: gc.close()
    probs=torch.softmax(logits,1)[0].cpu().numpy(); idx=int(probs.argmax())
    c1,c2=st.columns(2)
    c1.image(image,caption="Uploaded image",use_container_width=True)
    c2.metric("Predicted class",ck["class_names"][idx])
    c2.metric("Model score",f"{probs[idx]*100:.2f}%")
    c2.caption("Not a calibrated clinical probability.")
    st.image(overlay(image,cam),caption="Grad-CAM visualization",use_container_width=True)
    st.subheader("Scores by class")
    st.bar_chart({name:float(probs[i]) for i,name in enumerate(ck["class_names"])})
