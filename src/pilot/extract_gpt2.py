import argparse, numpy as np, torch
from transformers import AutoTokenizer, AutoModel
ap=argparse.ArgumentParser()
ap.add_argument("--in_txt",default="data/pilot/text_trs.txt")
ap.add_argument("--out_npy",default="data/pilot/X.npy")
ap.add_argument("--model",default="gpt2"); ap.add_argument("--layer",type=int,default=6)
ap.add_argument("--bs",type=int,default=4); a=ap.parse_args()
lines=[l.strip() for l in open(a.in_txt) if l.strip()]
tok=AutoTokenizer.from_pretrained(a.model)
if tok.pad_token is None: tok.pad_token=tok.eos_token
mdl=AutoModel.from_pretrained(a.model); mdl.config.pad_token_id=tok.pad_token_id; mdl.eval()
feats=[]
with torch.no_grad():
  for i in range(0,len(lines),a.bs):
    inp=tok(lines[i:i+a.bs],return_tensors="pt",padding=True,truncation=True,max_length=64)
    out=mdl(**inp,output_hidden_states=True); hs=out.hidden_states[a.layer]
    m=inp["attention_mask"].unsqueeze(-1); feats.append(((hs*m).sum(1)/m.sum(1)).cpu().numpy())
np.save(a.out_npy,np.vstack(feats).astype("float32")); print("saved",a.out_npy)
