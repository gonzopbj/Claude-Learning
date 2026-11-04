"""Identifiability pre-check: how much one-step variance of each child does each true parent explain?
Fits linear regressions of s_i(t+1)-s_i(t) on the full previous state vs. with object j removed."""
import numpy as np
from sklearn.linear_model import LinearRegression
from env import rollout, TRUE_PARENTS, N_OBJ
rng=np.random.default_rng(0)
S,A=rollout(400,60,rng)
X=np.concatenate([S[:,:-1].reshape(-1,N_OBJ*4),A.reshape(-1,2)],1)
for i in range(N_OBJ):
    Y=(S[:,1:,i]-S[:,:-1,i]).reshape(-1,4)
    full=LinearRegression().fit(X,Y).score(X,Y)
    row=[]
    for j in range(N_OBJ+1):
        cols=[c for c in range(X.shape[1]) if (c//4!=j if j<N_OBJ else c<N_OBJ*4)]
        r=LinearRegression().fit(X[:,cols],Y).score(X[:,cols],Y)
        row.append(full-r)
    print(f"child {i}: R2_full={full:.3f}  loss when dropping " + " ".join(f"{('a' if j==N_OBJ else j)}:{d:.3f}" for j,d in enumerate(row)))
print("positions range", S[...,:2].min(), S[...,:2].max(), "speed std per obj", S[...,2:].std(axis=(0,1,3)).round(3))
