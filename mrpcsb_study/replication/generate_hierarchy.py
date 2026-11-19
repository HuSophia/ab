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
    # Read the pinned Exponax 0.1.0 Fourier RHS directly. At the zero
    # state, velocity and vorticity gradients vanish, so the nonlinear
    # operator returns just the forcing. No time-integrator calibration.
    def rhs(injection_scale):
        case = scenario_type(**(params | {'injection_scale':injection_scale}))
        stepper = case.get_ref_stepper()
        sub = stepper.stepper if params['num_substeps'] > 1 else stepper
        nonlinear = sub._integrator._nonlinear_fun
        empty = jnp.zeros((1,n,n//2+1),dtype=jnp.complex64)
        hat = np.array(nonlinear(empty))[0]
        return np.fft.irfft2(hat,s=(n,n),axes=(-2,-1))
    field = rhs(params['injection_scale'])
    off = rhs(0.)
    assert np.linalg.norm(field)>0 and not np.any(off)
    # Independent check against the released solver's injection convention:
    # -injection_mode * injection_scale * cos(2*pi*injection_mode*y/L).
    phase = 2*np.pi*params['injection_mode']*np.arange(n)/n
    expected = -params['injection_mode']*params['injection_scale']*np.cos(phase)
    np.testing.assert_allclose(field,np.broadcast_to(expected,(n,n)),rtol=2e-6,atol=2e-6)
    forced = coordinates(field,waves)
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
    cli.add_argument('--train-seed', type=int, default=49123)
    cli.add_argument('--test-seed', type=int, default=89391)
    cli.add_argument('--endpoint-shuffle-seed', type=int, default=9821)
    a=cli.parse_args()
    if a.train_seed == a.test_seed:
        cli.error('Train and test generation must have distinct seeds.')
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
                train_seed=a.train_seed,test_seed=a.test_seed)
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
    rng=np.random.default_rng(a.endpoint_shuffle_seed)
    np.savez_compressed(a.out/'input.npz',source=tr[rng.permutation(ntr),0],
                        target=tr[rng.permutation(ntr),-1], test_paths=te,
                        test_times=finite*params['dt'],rates=rates,sigma=np.sqrt(2*rates*v),
                        forcing=force,waves=np.repeat(waves,2,axis=0),mu=mu,num_points=n)
    info=dict(profile=a.profile,scenario=scenario.get_scenario_name(),parameters=params,
              independent_evaluation_groups=groups,trajectories_per_group=group_n,
              endpoint_shuffle_seed=a.endpoint_shuffle_seed,
              retained_real_dimensions=len(force),forced_wave=waves[np.argmax(np.linalg.norm(force.reshape(-1,2),axis=1))].tolist(),
              seconds=time.perf_counter()-begin)
    (a.out/'metadata.json').write_text(json.dumps(info,indent=2)+'\n')
    print('Generated inputs for all prespecified evaluation groups.',flush=True)


if __name__=='__main__':
    main()
