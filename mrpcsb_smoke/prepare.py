"""Generate the official APEBench physical Kolmogorov scenario and export SB inputs."""
import argparse, json, time
from pathlib import Path
import importlib.metadata as md
import numpy as np

def fourier(v, cutoff):
    n=v.shape[-1]
    k=np.rint(np.fft.fftfreq(n)*n).astype(int)
    f=np.fft.fft2(v,axes=(-2,-1),norm='ortho')
    coords=[]; waves=[]
    for i,ki in enumerate(k):
        for j,kj in enumerate(k):
            if (i,j)==(0,0) or max(abs(ki),abs(kj))>=cutoff:continue
            if (i,j)>=((-i)%n,(-j)%n):continue
            coords += [np.sqrt(2)*f[...,i,j].real,np.sqrt(2)*f[...,i,j].imag]
            waves += [(ki,kj),(ki,kj)]
    return np.stack(coords,axis=-1),np.array(waves,int)

def export(train,test,settings,out,cutoff=2):
    n=settings['num_points']; dt=settings['dt']
    ids=np.array([0,2,4,6,8,10])
    tr,waves=fourier(train[:,[0,-1]],cutoff); te,_=fourier(test[:,ids],cutoff)
    avg=tr[:,0].mean(0); var=tr[:,0].var(0)
    var=np.maximum(var,max(np.median(var)*1e-4,1e-12))
    rates=settings['diffusivity']*(2*np.pi/settings['domain_extent'])**2*(waves**2).sum(1)-settings['drag']
    rng=np.random.default_rng(91)
    np.savez_compressed(out/'bridge_input.npz',source=tr[rng.permutation(len(tr)),0]-avg,
        target=tr[rng.permutation(len(tr)),-1]-avg,rates=rates,sigma=np.sqrt(2*rates*var),
        T=10*dt,mu=avg,waves=waves,num_points=n,train_ids=np.arange(len(tr)),
        test_ids=np.arange(len(te))+len(tr),test_paths=te-avg,test_times=ids*dt)
    np.savez_compressed(out/'test_snapshots.npz',fields=test[:,ids].astype('float32'),
        times=ids*dt,trajectory_id=np.arange(len(te))+len(tr))

def main():
    import apebench, jax
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=argparse.ArgumentParser();p.add_argument('--out',default='ape_run')
    p.add_argument('--profile',choices=['smoke','study'],default='smoke');a=p.parse_args()
    path=Path(a.out);path.mkdir(parents=True,exist_ok=False)
    n,tr,te,warm=(64,16,16,20) if a.profile=='smoke' else (128,128,64,500)
    # Physical constants as specified for the official KolmogorovFlow scenario.
    params=dict(num_spatial_dims=2,num_points=n,domain_extent=2*np.pi,dt=.1,
        diffusivity=.01,drag=-.1,injection_mode=4,injection_scale=1.,num_substeps=20,
        order=2,dealiasing_fraction=2/3,num_warmup_steps=warm,
        num_train_samples=tr,num_test_samples=te,train_temporal_horizon=10,test_temporal_horizon=10,
        train_seed=971,test_seed=2573)
    scenario=apebench.scenarios.physical.KolmogorovFlow(**params)
    prev=time.perf_counter()
    def check(v,count):
        v=np.asarray(v)
        assert v.shape==(count,11,1,n,n),(v.shape,count,n)
        assert np.isfinite(v).all()
        return v[:,:,0]
    train=check(scenario.get_train_data(),tr);test=check(scenario.get_test_data(),te)
    sec=time.perf_counter()-prev
    export(train,test,params,path)
    meta=dict(scenario=scenario.get_scenario_name(),profile=a.profile,
        parameters=params,generation_seconds_including_jit=sec,
        versions={k:md.version(k) for k in ('apebench','jax','exponax','numpy')},
        devices=[str(d) for d in jax.devices()],scope="APEBench-generated unpaired snapshot interpolation; not the official forecasting protocol")
    (path/'metadata.json').write_text(json.dumps(meta,indent=2))
    idx=[0,2,4,6,8,10];fig,axs=plt.subplots(2,len(idx),figsize=(12,4.7),layout='constrained')
    lim=float(np.quantile(np.abs(train),.99))
    for i,row in enumerate(axs):
        for j,k in enumerate(idx):
            ax=row[j]; im=ax.imshow(test[i,k],origin='lower',cmap='RdBu_r',vmin=-lim,vmax=lim)
            if i==0:ax.set_title(f't={k*params["dt"]:.1f}')
            ax.set(xticks=[],yticks=[])
    fig.colorbar(im,ax=axs,label='Vorticity',shrink=.8);fig.suptitle("APEBench Kolmogorov flow: held-out reference fields")
    for ext in ['pdf','png']:fig.savefig(path/f'apebench_reference.{ext}',dpi=180)
    print(json.dumps(meta,indent=2))

if __name__=='__main__':main()
