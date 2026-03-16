import argparse, json, pathlib, numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
def mean_corr(y,yh):
  y=y-y.mean(0); yh=yh-yh.mean(0)
  return float(np.mean((y*yh).sum(0)/np.sqrt((y*y).sum(0)*(yh*yh).sum(0)+1e-9)))
ap=argparse.ArgumentParser()
ap.add_argument("--X",default="data/pilot/X.npy"); ap.add_argument("--Y",default="data/pilot/Y.npy")
ap.add_argument("--out_dir",default="artifacts/train"); ap.add_argument("--splits",type=int,default=5)
a=ap.parse_args(); X=np.load(a.X); Y=np.load(a.Y)
alphas=np.logspace(2,7,16); cv=TimeSeriesSplit(n_splits=a.splits); scores=[]
for alpha in alphas:
  fold=[]; 
  for tr,te in cv.split(X): m=Ridge(alpha=alpha).fit(X[tr],Y[tr]); fold.append(mean_corr(Y[te],m.predict(X[te])))
  scores.append(np.mean(fold))
best=float(alphas[int(np.argmax(scores))]); m=Ridge(alpha=best).fit(X,Y); Yhat=m.predict(X)
out=pathlib.Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
np.save(out/"Yhat.npy",Yhat); json.dump({"best_alpha":best,"cv_mean_corr":float(max(scores))}, open(out/"train_metrics.json","w"))
print("best_alpha",best,"saved",out/"Yhat.npy")
