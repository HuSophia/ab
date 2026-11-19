"""Independent tiny-case primal QA for block-balancing certificates."""
import numpy as np
from scipy.special import xlogy
from scipy.linalg import qr
from scipy.optimize import LinearConstraint, minimize
from eval_hierarchy import coarse_sinkhorn

def constrained(k,a,b,I,J,q,start):
    idmat=np.arange(k.size).reshape(k.shape)
    rows=list(np.eye(k.shape[0]).repeat(k.shape[1],axis=1))
    rows+=list(np.tile(np.eye(k.shape[1]),(1,k.shape[0])))
    for ix in I:
        for iy in J:
            row=np.zeros(k.size)
            row[idmat[np.ix_(ix,iy)].ravel()]=1
            rows.append(row)
    A=np.array(rows)
    v=np.r_[a,b,q.ravel()]
    _,R,piv=qr(A.T,pivoting=True,mode='economic')
    use=piv[:np.count_nonzero(np.abs(R.diagonal())>1e-10)]
    kernel=k.ravel()
    obj=lambda z: np.sum(xlogy(z,z) - z*kernel)
    grad=lambda z: np.log(z)+1-kernel
    l=LinearConstraint(A[use],v[use],v[use])
    r=minimize(obj,start.ravel(),jac=grad,bounds=[(1e-12,None)]*k.size,constraints=l,
               method='SLSQP',options={'ftol':1e-13,'maxiter':2000})
    if not r.success:
        raise RuntimeError(r.message)
    np.testing.assert_allclose(A@r.x,v,atol=1e-10)
    return r.x.reshape(k.shape)

def main():
    x=np.array([-2,-1,1,2])
    K=-.5*(x[None]-0.7*x[:,None])**2
    a=b=np.full(4,.25)
    I=J=[np.arange(2),np.arange(2,4)]
    P=coarse_sinkhorn(K,a,b)[0]
    Q=np.array([[.4,.1],[.1,.4]])
    L=np.zeros_like(K)
    for ai,i in enumerate(I):
        for bj,j in enumerate(J):
            L[np.ix_(i,j)]=Q[ai,bj]*coarse_sinkhorn(K[np.ix_(i,j)],np.full(2,.5),np.full(2,.5))[0]
    R=constrained(K,a,b,I,J,Q,L)
    kl=lambda u,v:np.sum(xlogy(u,u)-xlogy(u,v))
    f=lambda u: np.sum(xlogy(u,u) - u*K)
    # Objective ordering and equality up to numerical precision.
    np.testing.assert_allclose(f(L)-f(R),kl(L,R),atol=1e-7,rtol=1e-6)
    np.testing.assert_allclose(f(R)-f(P),kl(R,P),atol=1e-9,rtol=1e-6)
    print(f'Proportional-lift penalty: {f(L)-f(R):.8f}; coarse allocation penalty: {f(R)-f(P):.8f}')
    Q0=np.array([[P[np.ix_(i,j)].sum() for j in J] for i in I])
    R0=constrained(K,a,b,I,J,Q0,P)
    np.testing.assert_allclose(R0,P,atol=1e-8)
    print('Projected-flat oracle recovers the flat plan.')

if __name__=='__main__':
    main()
