"""Unpaired diagonal-OU SB drift regression. Small pilot, not full-paper evidence.
Input npz: source[N,D], target[M,D], rates[D], sigma[D], optional T scalar.
Endpoint rows are independently sampled; no row pairing is used.
Full diagonal OU conditional laws, endpoint Sinkhorn and stochastic Euler rollouts.
Hierarchy uses all block pairs and is dense; no scalability claim.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from scipy.cluster.vq import kmeans2


def variance(t,r,s):
    t=np.asarray(t); rr=np.maximum(r,1e-12)
    return np.where(r>1e-10,s*s*(-np.expm1(-2*rr*t))/(2*rr),s*s*t)


SOLVER_AUDIT=[]
def sinkhorn(K,a,b,tol=1e-7):
    # Continuation solves the original final kernel. Capped iterative result is
    # made exactly feasible using nonnegative row/column scaling and rank-one
    # marginal-deficit correction (Altschuler et al., 2017 rounding algorithm).
    f=np.zeros(len(a));g=np.zeros(len(b));start=max(1.,float(np.ptp(K))/5)
    for temp in np.geomspace(start,1.,8):
        L=K/temp;u=f/temp;v=g/temp
        for it in range(5000):
            u=np.log(a)-logsumexp(L+v[None],axis=1)
            v=np.log(b)-logsumexp(L+u[:,None],axis=0)
            if it%25==0:
                P=np.exp(L+u[:,None]+v[None]);err=max(abs(P.sum(1)-a).max(),abs(P.sum(0)-b).max())
                if err<(tol if temp==1 else 1e-5):break
        f=u*temp;g=v*temp
    P=np.exp(K+u[:,None]+v[None]);before=P.copy()
    raw=max(abs(P.sum(1)-a).max(),abs(P.sum(0)-b).max())
    P*=np.minimum(a/np.maximum(P.sum(1),1e-300),1)[:,None]
    P*=np.minimum(b/np.maximum(P.sum(0),1e-300),1)[None,:]
    da=np.maximum(a-P.sum(1),0);db=np.maximum(b-P.sum(0),0)
    if da.sum()>1e-15 and db.sum()>1e-15:P+=np.outer(da,db)/da.sum()
    err=max(abs(P.sum(1)-a).max(),abs(P.sum(0)-b).max())
    if err>1e-12:raise RuntimeError(f'Rounded Sinkhorn feasibility failed: {err}')
    change=float(abs(P-before).sum())
    # Rounding restores feasibility, not exact entropic optimality.
    objective=lambda X:float(np.sum(np.where(X>0,X*(np.log(np.maximum(X,1e-300))-K),0)))
    SOLVER_AUDIT.append(dict(raw_residual=float(raw),rounded_residual=float(err),rounding_L1=change,objective_change=objective(P)-objective(before)))
    return P


def coupling(x,y,r,s,T,kind,k,seed):
    q=variance(T,r,s)
    K=-.5*np.sum((x[:,None]*np.exp(-r*T)-y[None])**2/q,axis=2)
    a=np.ones(len(x))/len(x);b=np.ones(len(y))/len(y)
    if kind=='flat':return sinkhorn(K,a,b)
    _,lx=kmeans2(x,min(k,len(x)),minit='points',seed=seed)
    _,ly=kmeans2(y,min(k,len(y)),minit='points',seed=seed+1)
    I=[np.flatnonzero(lx==i) for i in np.unique(lx)];J=[np.flatnonzero(ly==j) for j in np.unique(ly)]
    C=np.array([[logsumexp(K[np.ix_(i,j)])-np.log(len(i)*len(j)) for j in J] for i in I])
    macro=sinkhorn(C,np.array([len(i)/len(x) for i in I]),np.array([len(j)/len(y) for j in J]))
    P=np.zeros_like(K)
    for ii,i in enumerate(I):
        for jj,j in enumerate(J):P[np.ix_(i,j)]=macro[ii,jj]*sinkhorn(K[np.ix_(i,j)],np.ones(len(i))/len(i),np.ones(len(j))/len(j))
    assert max(abs(P.sum(1)-a).max(),abs(P.sum(0)-b).max())<1e-6
    return P


def targets(x,y,P,r,s,T,n,rng,eps):
    ij=rng.choice(P.size,n,p=(P/P.sum()).ravel());i,j=np.unravel_index(ij,P.shape)
    t=rng.uniform(eps*T,(1-eps)*T,(n,1));h=T-t
    qt=variance(t,r,s);qh=variance(h,r,s);qT=variance(T,r,s)
    m=x[i]*np.exp(-r*t)+(qt*np.exp(-r*h)/qT)*(y[j]-x[i]*np.exp(-r*T))
    var=qt*qh/qT
    z=m+np.sqrt(var)*rng.normal(size=m.shape)
    drift=-r*z+s*s*np.exp(-r*h)/qh*(y[j]-np.exp(-r*h)*z)
    return t,z,drift


class NumpyMLP:
    def __init__(self,d,width,seed):
        rng=np.random.default_rng(seed);dims=[d+1,width,width,d]
        self.p=[]
        for a,b in zip(dims[:-1],dims[1:]):self.p.extend([rng.normal(size=(a,b))/np.sqrt(a),np.zeros(b)])
        self.m=[np.zeros_like(p) for p in self.p];self.v=[p.copy() for p in self.m];self.step=0
    def forward(self,x):
        h1=np.tanh(x@self.p[0]+self.p[1]);h2=np.tanh(h1@self.p[2]+self.p[3]);return h2@self.p[4]+self.p[5],(x,h1,h2)
    def predict(self,x):return self.forward(x)[0]
    def update(self,x,y,lr):
        out,(x,h1,h2)=self.forward(x);err=out-y;g=2*err/err.size
        grads=[None]*6;grads[4]=h2.T@g;grads[5]=g.sum(0)
        g=(g@self.p[4].T)*(1-h2*h2);grads[2]=h1.T@g;grads[3]=g.sum(0)
        g=(g@self.p[2].T)*(1-h1*h1);grads[0]=x.T@g;grads[1]=g.sum(0)
        norm=np.sqrt(sum(np.sum(g*g) for g in grads));clip=min(1.,10/max(norm,1e-12));self.step+=1
        for i,(p,g) in enumerate(zip(self.p,grads)):
            g=g*clip;self.m[i]=.9*self.m[i]+.1*g;self.v[i]=.999*self.v[i]+.001*g*g
            p-=lr*(self.m[i]/(1-.9**self.step))/(np.sqrt(self.v[i]/(1-.999**self.step))+1e-8)
        return float(np.mean(err*err))
    def save(self,path):np.savez_compressed(path,**{f'p{i}':p for i,p in enumerate(self.p)})


class TorchMLP:
    def __init__(self,d,width,seed):
        import torch
        self.torch=torch;torch.manual_seed(seed);self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.net=torch.nn.Sequential(torch.nn.Linear(d+1,width),torch.nn.Tanh(),torch.nn.Linear(width,width),torch.nn.Tanh(),torch.nn.Linear(width,d)).to(self.device)
        self.opt=torch.optim.Adam(self.net.parameters(),lr=1e-3)
    def tensor(self,x):return self.torch.tensor(x,dtype=self.torch.float32,device=self.device)
    def predict(self,x):
        with self.torch.no_grad():return self.net(self.tensor(x)).cpu().numpy()
    def update(self,x,y,lr):
        for group in self.opt.param_groups:group['lr']=lr
        loss=(self.net(self.tensor(x))-self.tensor(y)).square().mean();self.opt.zero_grad();loss.backward();self.torch.nn.utils.clip_grad_norm_(self.net.parameters(),10);self.opt.step();return loss.item()
    def save(self,path):self.torch.save(self.net.state_dict(),str(path)+'.pt')


def sw2(x,y,seed):
    rg=np.random.default_rng(seed);v=rg.normal(size=(x.shape[1],64));v/=np.linalg.norm(v,axis=0)
    return float(np.mean((np.sort(x@v,axis=0)-np.sort(y@v,axis=0))**2))


def run(args):
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    with np.load(args.input,allow_pickle=False) as z:
        x=z['source'];y=z['target'];rates=z['rates'];sigma=z['sigma'];T=float(z['T']) if 'T' in z else 1.
        test_paths=z['test_paths'];test_times=z['test_times'];test_ids=z['test_ids'];train_ids=z['train_ids']
    assert x.ndim==y.ndim==2 and x.shape[1]==y.shape[1]==len(rates)==len(sigma)
    assert np.all(sigma>0) and np.all(rates>=0)
    scale=np.maximum(np.std(np.concatenate([x,y]),axis=0),1e-4);scale=np.maximum(scale,.05*np.median(scale))
    # Scaling uses endpoints only; preserve physical zero and transform diffusion.
    x=x/scale;y=y/scale;sigma=sigma/scale;test_paths=test_paths/scale
    backend=args.backend
    if backend=='auto':
        try:import torch;backend='torch'
        except ImportError:backend='numpy'
    cls=TorchMLP if backend=='torch' else NumpyMLP
    rows=[]; eval_rows=[];cache={}
    # Freeze all four endpoint couplings and hierarchy partitions across training seeds.
    for ref in ['brownian','physical']:
        r=np.zeros_like(rates) if ref=='brownian' else rates
        for kind in ['flat','hierarchical']:
            start=time.perf_counter();begin=len(SOLVER_AUDIT);P=coupling(x,y,r,sigma,T,kind,args.clusters,0);ct=time.perf_counter()-start
            cache[ref,kind]=(P,ct);np.savez_compressed(out/f'coupling_{ref}_{kind}.npz',coupling=P)
            (out/f'solver_{ref}_{kind}.json').write_text(json.dumps(SOLVER_AUDIT[begin:],indent=2))
    for seed in range(args.seeds):
      for ref in ['brownian','physical']:
       r=np.zeros_like(rates) if ref=='brownian' else rates
       for kind in ['flat','hierarchical']:
        P,ct=cache[ref,kind]
        model=cls(x.shape[1],args.width,seed);rng=np.random.default_rng(seed+100);losses=[]
        def features(t,z):return np.concatenate([t/T,z],axis=1)
        # Fit control residual relative to known reference, shared MLP architecture.
        vt,vz,vb=targets(x,y,P,r,sigma,T,args.batch,np.random.default_rng(seed+998),args.eps)
        before=float(np.mean((model.predict(features(vt,vz))-(vb+r*vz))**2))
        start=time.perf_counter()
        for it in range(args.updates):
            t,z,b=targets(x,y,P,r,sigma,T,args.batch,rng,args.eps)
            losses.append(model.update(features(t,z),b+r*z,args.lr))
        after=float(np.mean((model.predict(features(vt,vz))-(vb+r*vz))**2));training=time.perf_counter()-start
        start=time.perf_counter();rg=np.random.default_rng(seed+10000);z=x[rg.integers(len(x),size=args.particles)].copy();paths=[z.copy()];dt=T/args.steps
        for j in range(args.steps):
            t=np.full((len(z),1),np.clip(j*dt,args.eps*T,(1-args.eps)*T));z+=dt*(-r*z+model.predict(features(t,z)))+sigma*np.sqrt(dt)*rg.normal(size=z.shape)
            if not np.isfinite(z).all():raise RuntimeError('Nonfinite rollout')
            paths.append(z.copy())
        sampling=time.perf_counter()-start
        obs=y[rg.integers(len(y),size=len(z))];name=f'{ref}_{kind}_seed{seed}'
        np.savez_compressed(out/(name+'.npz'),paths=np.array(paths)*scale,times=np.linspace(0,T,args.steps+1),coupling=P,loss=np.array(losses),scale=scale)
        # Independent heldout initial populations. Intermediate DNS times were
        # withheld entirely from training and reference calibration.
        eval_start=time.perf_counter()
        evalpaths=[]
        for draw in range(3):
            rg=np.random.default_rng(700000+seed*100+draw)
            zz=test_paths[rg.integers(len(test_paths),size=args.particles),0].copy()
            predicted=[zz.copy()]
            for ti in range(1,len(test_times)):
                left=float(test_times[ti-1]);right=float(test_times[ti]);ns=max(1,int(np.ceil((right-left)/T*args.steps)));dt2=(right-left)/ns
                for j in range(ns):
                    tt=np.full((len(zz),1),np.clip(left+j*dt2,args.eps*T,(1-args.eps)*T))
                    zz+=dt2*(-r*zz+model.predict(features(tt,zz)))+sigma*np.sqrt(dt2)*rg.normal(size=zz.shape)
                predicted.append(zz.copy())
                refsample=test_paths[rg.integers(len(test_paths),size=len(zz)),ti]
                refsample2=test_paths[rg.integers(len(test_paths),size=len(zz)),ti]
                eval_rows.append(dict(seed=seed,reference=ref,coupling=kind,draw=draw,t=right,region='endpoint' if ti==len(test_times)-1 else 'interior',SW2_squared=sw2(zz,refsample,123),empirical_resampling_variability=sw2(refsample,refsample2,123)))
            if draw==0:evalpaths=np.stack(predicted)
        np.savez_compressed(out/(name+'_heldout_paths.npz'),marginals=evalpaths*scale,times=test_times,test_ids=test_ids,train_ids=train_ids)
        eval_seconds=time.perf_counter()-eval_start
        model.save(out/(name+'_weights'))
        rows.append(dict(seed=seed,reference=ref,coupling=kind,backend=backend,updates=args.updates,heldout_conditional_target_MSE_before=before,heldout_conditional_target_MSE_after=after,endpoint_SW2_squared=sw2(z,obs,seed),max_marginal_residual=float(max(abs(P.sum(1)-1/len(x)).max(),abs(P.sum(0)-1/len(y)).max())),coupling_seconds=ct,training_seconds=training,sampling_seconds=sampling,heldout_eval_seconds=eval_seconds,total_seconds=ct+training+sampling+eval_seconds))
        print(rows[-1],flush=True)
    (out/'results.json').write_text(json.dumps(rows,indent=2));(out/'heldout_metrics.json').write_text(json.dumps(eval_rows,indent=2));(out/'config.json').write_text(json.dumps(vars(args),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--out',required=True);p.add_argument('--backend',choices=['auto','torch','numpy'],default='auto')
    for name,value in [('updates',200),('batch',64),('width',32),('particles',128),('steps',100),('seeds',1),('clusters',4)]:p.add_argument('--'+name,type=int,default=value)
    p.add_argument('--lr',type=float,default=.001);p.add_argument('--eps',type=float,default=.02);run(p.parse_args())
