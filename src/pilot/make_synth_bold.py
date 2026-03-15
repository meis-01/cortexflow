import argparse, pathlib, json, numpy as np
ap=argparse.ArgumentParser()
ap.add_argument("--X",default="data/pilot/X.npy"); ap.add_argument("--out_dir",default="data/pilot")
ap.add_argument("--n_vox",type=int,default=200); ap.add_argument("--snr",type=float,default=2.0)
ap.add_argument("--seed",type=int,default=0); a=ap.parse_args()
rng=np.random.default_rng(a.seed); X=np.load(a.X).astype("float32")
W=rng.standard_normal((X.shape[1],a.n_vox)).astype("float32")
Y=X@W + rng.standard_normal((X.shape[0],a.n_vox)).astype("float32")/a.snr
Y=(Y-Y.mean(0))/(Y.std(0)+1e-6)
out=pathlib.Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
np.save(out/"Y.npy",Y); np.save(out/"W_true.npy",W)
json.dump({"n_tr":int(Y.shape[0]),"n_vox":int(a.n_vox)}, open(out/"synth_meta.json","w"))
print("saved",out/"Y.npy")
