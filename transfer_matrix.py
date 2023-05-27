from opt_einsum import contract # idk why but its required to avoid bug in contract with numpy arrays
import torch
import numpy as np
from tqdm.auto import tqdm
from scipy.sparse.linalg import LinearOperator,eigs


def wrap_pytorch(func):
    return lambda v:func(torch.tensor(v)).detach().cpu().numpy()

def wrap_pbar(pbar):
    return lambda func: lambda *args, **kwargs: (func(*args, **kwargs), pbar.update(1))[0]

def get_transfer_matrix_operator_2D(T,n=2):
    bond_dim=T.shape[0]
    bond_dim1=T.shape[2]
    pbar=tqdm(leave=False)
    @wrap_pbar(pbar)
    @wrap_pytorch
    def matvec(v):
        #     1 2 3
        #(-3)-T-|-|-(-4)
        #     [ v ] 
        v=contract('ij,...->ij...',torch.eye(bond_dim1),v.reshape((bond_dim,)*n))
        for i in range(n):
            idx1=[-3,-5]+list(-2 if j==i else j for j in range(n))
            idx2=[-1,-2,-5,-4]
            idx3=[-3,-4]+list(-1 if j==i else j for j in range(n))
            v=contract(v,idx1,T,idx2,idx3)
        return contract('ii...->...',v).flatten()
    @wrap_pbar(pbar)
    @wrap_pytorch
    def rmatvec(v):
        v=contract('ij,...->ij...',torch.eye(bond_dim1),v.reshape((bond_dim,)*n))
        for i in range(n):
            idx1=[-3,-5]+list(-1 if j==i else j for j in range(n))
            idx2=[-1,-2,-5,-4]
            idx3=[-3,-4]+list(-2 if j==i else j for j in range(n))
            v=contract(v,idx1,T.conj(),idx2,idx3)
        return contract('ii...->...',v).flatten()
    return LinearOperator(shape=(bond_dim**n,bond_dim**n),matvec=matvec,rmatvec=rmatvec)

def get_transfer_matrix_operator_3D(T,n=(2,2)):
    bond_dim=T.shape[0]
    pbar=tqdm(leave=False)
    @wrap_pbar(pbar)
    @wrap_pytorch
    def matvec(v):
        if n==(2,2):
            return contract('iIabxy,jJbazw,kKcdyx,lLdcwz,IJKL->ijkl',T,T,T,T,v).flatten()
    @wrap_pbar(pbar)
    @wrap_pytorch
    def rmatvec(v):
        if n==(2,2):
            return contract('iIabxy,jJbazw,kKcdyx,lLdcwz,ijkl->IJKL',T,T,T,T,v.conj()).conj().flatten()
    if n not in [(2,2)]:
        raise NotImplementedError
    return LinearOperator(shape=(bond_dim**np.prod(n),bond_dim**np.prod(n)),matvec=matvec,rmatvec=rmatvec)


def get_transfer_matrix_operator(T,n:'tuple[int]'):
    if len(T.shape)==4:
        assert len(n)==1
        M=get_transfer_matrix_operator_2D(T,n[0])
    elif len(T.shape)==6:
        assert len(n)==2
        if n[1]==1:
            M=get_transfer_matrix_operator_2D(contract('ijklmm->ijkl',T),n[0])
        elif n[0]==1:
            M=get_transfer_matrix_operator_2D(contract('ijmmkl->ijkl',T),n[1])
        else:
            M=get_transfer_matrix_operator_3D(T,n)
    else:
        raise NotImplementedError
    return M


def get_scdims(T,n=2,k=10,tensor_block_height=1):
    if isinstance(n,int): n=(n,) if len(T.shape)==4 else (n,n)
    M=get_transfer_matrix_operator(T,n)
    s,u=eigs(M,k=min(k,M.shape[0]-2))
    u,s=torch.tensor(u),torch.tensor(s)
    s,u=s.abs()[s.abs().argsort(descending=True)],u[:,s.abs().argsort(descending=True)]
    n_scaling=np.prod(n)**(1/len(n))
    scaling=np.exp(2*np.pi/n_scaling*tensor_block_height)
    scdims=torch.log(s/s[0]).abs()/torch.log(torch.as_tensor(scaling))
    eigvecs=u.T
    return scdims,eigvecs