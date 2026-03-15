import argparse, glob, pathlib, numpy as np
from nilearn.maskers import NiftiMasker
ap=argparse.ArgumentParser()
ap.add_argument("--root",default="data/openneuro/ds003643"); ap.add_argument("--out_dir",default="data/real")
ap.add_argument("--max_vox",type=int,default=500); a=ap.parse_args()
bold=sorted(glob.glob(f"{a.root}/sub-*/func/*_bold.nii*"))[0]
Y=NiftiMasker(standardize=True,detrend=True).fit_transform(bold)[:, :a.max_vox].astype("float32")
out=pathlib.Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
np.save(out/"Y.npy",Y); (out/"text_trs.txt").write_text("\n".join(["placeholder"]*Y.shape[0])+"\n")
print("wrote",Y.shape,"from",bold)
