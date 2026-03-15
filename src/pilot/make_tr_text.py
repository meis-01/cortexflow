import argparse, pathlib, random
ap=argparse.ArgumentParser()
ap.add_argument("--out",default="data/pilot/text_trs.txt")
ap.add_argument("--n_tr",type=int,default=120)
ap.add_argument("--seed",type=int,default=0)
a=ap.parse_args(); random.seed(a.seed)
base=("The little prince asked questions. He met a fox and learned about taming. "
      "Stars are beautiful because of a flower.")
lines=[base if random.random()>0.2 else base[::-1] for _ in range(a.n_tr)]
p=pathlib.Path(a.out); p.parent.mkdir(parents=True,exist_ok=True)
p.write_text("\n".join(lines)+"\n")
print("wrote",len(lines),"lines to",p)
