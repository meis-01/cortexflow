import argparse, json, pathlib, numpy as np, matplotlib.pyplot as plt
def corr(y,yh):
  y=y-y.mean(0); yh=yh-yh.mean(0)
  return (y*yh).sum(0)/np.sqrt((y*y).sum(0)*(yh*yh).sum(0)+1e-9)
ap=argparse.ArgumentParser()
ap.add_argument("--Y",default="data/pilot/Y.npy"); ap.add_argument("--Yhat",default="artifacts/train/Yhat.npy")
ap.add_argument("--out_dir",default="artifacts/eval"); a=ap.parse_args()
Y=np.load(a.Y); Yhat=np.load(a.Yhat); r=corr(Y,Yhat).astype("float32")
out=pathlib.Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
np.save(out/"corr_per_voxel.npy",r); json.dump({"mean_r":float(r.mean())}, open(out/"eval_metrics.json","w"))
plt.hist(r,bins=30); plt.xlabel("Pearson r"); plt.ylabel("count"); plt.tight_layout()
plt.savefig(out/"corr_hist.png",dpi=150); print("mean_r",float(r.mean()))
