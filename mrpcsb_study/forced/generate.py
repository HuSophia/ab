"""Official APEBench forced-mode diagnostic; export train endpoints and fresh evaluation groups."""
import argparse
from pathlib import Path
import json
import time

import numpy as np


def coordinates(field, waves):
    n = field.shape[-1]
    fft = np.fft.fft2(field, axes=(-2, -1), norm='ortho')
    ix, iy = (waves.T % n)
    values = fft[..., ix, iy]
    return np.sqrt(2) * np.stack((values.real, values.imag), axis=-1).reshape(*field.shape[:-2], -1)


def wave_catalog(n):
    k = np.rint(np.fft.fftfreq(n)*n).astype(int)
    return np.array([(int(a),int(b)) for i,a in enumerate(k) for j,b in enumerate(k)
                     if (i,j) < ((-i)%n,(-j)%n) and
                     (max(abs(a), abs(b)) < 2 or sorted((abs(a),abs(b))) == [0,4])])


def extract_force(n, params, scenario_type, waves):
    import jax.numpy as jnp
    r = params['diffusivity'] * (2*np.pi/params['domain_extent'])**2 * np.sum(waves**2,axis=1) - params['drag']
    # A zero start evolves into a single Fourier harmonic. The vorticity
    # advection of that harmonic by its own velocity vanishes. Divide the
    # solver's one-step response by the linear accumulation factor.
    def get(dt, injection_scale):
        scenario=scenario_type(**(params | {'dt':dt,'injection_scale':injection_scale}))
        field=np.asarray(scenario.get_ref_stepper()(jnp.zeros((1,n,n),dtype=jnp.float32)))[0]
        coeff=coordinates(field,waves)
        accum=-np.expm1(-r*dt)/r
        return coeff/np.repeat(accum,2), float(np.sum(field.astype(float)**2))
    forced, norm = get(params['dt'], params['injection_scale'])
    half, _ = get(params['dt']/2, params['injection_scale'])
    off, off_norm = get(params['dt'], 0.)
    assert norm>0 and off_norm==0 and np.count_nonzero(off)==0
    assert np.linalg.norm(forced-half)/np.linalg.norm(forced)<1e-3, 'forcing failed step-halving check'
    pair=np.linalg.norm(forced.reshape(-1,2),axis=1)
    forced_pair=(pair>pair.max()*1e-5)
    assert forced_pair.sum()==1 and sorted(abs(waves[forced_pair][0]))==[0,4]
    keep=(np.max(abs(waves),axis=1)<2) | forced_pair
    waves=waves[keep]
    r=np.repeat(r[keep],2)
    coeff=forced[np.repeat(keep,2)]
    coeff[np.abs(coeff)<np.max(np.abs(coeff))*1e-6]=0
    ncoord=coeff.size
    assert ncoord==10 and np.count_nonzero(coeff) in (1,2)
    return waves,r,coeff


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--out',type=Path,required=True)
    cli.add_argument('--profile',choices=('smoke','study'),default='study')
    a=cli.parse_args()
    import apebench
    a.out.mkdir(parents=True,exist_ok=False)
    is_smoke=a.profile=='smoke'
    n,ntr,group_n,warm,steps=(16,8,4,4,20) if is_smoke else (64,64,32,500,20)
    groups=5
    params=dict(num_spatial_dims=2,num_points=n,domain_extent=2*np.pi,dt=.1,
                diffusivity=.01,drag=-.1,injection_mode=4,injection_scale=1.,
                num_substeps=20,order=2,dealiasing_fraction=2/3,num_warmup_steps=warm,
                num_train_samples=ntr,num_test_samples=groups*group_n,
                train_temporal_horizon=steps,test_temporal_horizon=steps,
                train_seed=49123,test_seed=65477)
    case=apebench.scenarios.physical.KolmogorovFlow
    waves, rates, force=extract_force(n,params,case,wave_catalog(n))
    scenario=case(**params)
    begin=time.perf_counter()
    def get(z,num):
        a=np.asarray(z)
        assert a.shape==(num,steps+1,1,n,n) and np.isfinite(a).all()
        return a[:,:,0]
    train=get(scenario.get_train_data(),ntr)
    tr=coordinates(train[:,[0,-1]],waves)
    del train
    test=get(scenario.get_test_data(),groups*group_n)
    finite=np.arange(0,steps+1,4)
    te=coordinates(test[:,finite],waves).reshape(groups,group_n,len(finite),-1)
    del test
    mu=tr[:,0].mean(0)
    # Reuse the earlier scale convention; no held-out outcomes enter calibration.
    v=tr[:,0].var(0)
    v=np.maximum(v,max(np.median(v)*1e-4,1e-12))
    rng=np.random.default_rng(9821)
    np.savez_compressed(a.out/'input.npz',source=tr[rng.permutation(ntr),0],
                        target=tr[rng.permutation(ntr),-1], test_paths=te,
                        test_times=finite*params['dt'],rates=rates,sigma=np.sqrt(2*rates*v),
                        forcing=force,waves=np.repeat(waves,2,axis=0),mu=mu,num_points=n)
    info=dict(profile=a.profile,scenario=scenario.get_scenario_name(),parameters=params,
              independent_evaluation_groups=groups,trajectories_per_group=group_n,
              retained_real_dimensions=len(force),forced_wave=waves[np.argmax(np.linalg.norm(force.reshape(-1,2),axis=1))].tolist(),
              seconds=time.perf_counter()-begin)
    (a.out/'metadata.json').write_text(json.dumps(info,indent=2)+'\n')
    print('Generated inputs for all prespecified evaluation groups.',flush=True)


if __name__=='__main__':
    main()
