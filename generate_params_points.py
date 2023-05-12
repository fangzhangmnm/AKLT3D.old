import numpy as np
from TNModels import AKLT3D, Ising2D


# bash_filename='aklt3d_scan_params.sh'
# tensor_filename_template='data/aklt3d_scan_params/a1_%.3f_a2_%.3f_a3_%.3f.pt'

# da1_range=np.linspace(-1,1,11)
# da2_range=np.linspace(-1,1,11)
# da3_range=np.linspace(-1,1,11)

# params=AKLT3D.get_default_params() # 1.154 1.826 4.472
# a10,a20,a30=params['a1'],params['a2'],params['a3']

# with open(bash_filename,'w') as f:
#     for da1 in da1_range:
#         for da2 in da2_range:
#             for da3 in da3_range:
#                 a1=a10+da1
#                 a2=a20+da2
#                 a3=a30+da3
#                 params1={'a1':a1,'a2':a2,'a3':a3}
#                 params1=str(params1)
#                 f.write('python HOTRG_run.py --filename %s --nLayers 30 --max_dim 10 --gilt_enabled --mcf_enabled  --model AKLT3D --params "%s"\n'%(tensor_filename_template%(a1,a2,a3),params1))


# bash_filename_template='ising2d_gilt_X{bond_dim}_scan_params.sh'
# tensor_filename_template='data/ising2d_gilt_X{bond_dim}_scan_params/beta_{beta:.7f}.pt'
# command_template='python HOTRG_run.py --filename {tensor_filename} --nLayers {nLayers} --max_dim {bond_dim} --gilt_enabled --mcf_enabled  --model Ising2D --params "{params}" --device cuda:1\n'

bash_filename_template='ising2d_X{bond_dim}_scan_params.sh'
tensor_filename_template='data/ising2d_X{bond_dim}_scan_params/beta_{beta:.7f}.pt'
command_template='python HOTRG_run.py --filename {tensor_filename} --nLayers {nLayers} --max_dim {bond_dim} --mcf_enabled  --model Ising2D --params "{params}" --device cuda:1\n'


nLayers=60
for bond_dim in [12,14,16,18,20,22,24]:
    bash_filename=bash_filename_template.format(bond_dim=bond_dim)
    dbeta_range=np.linspace(-0.01,0.01,11)
    params=Ising2D.get_default_params()
    beta0=params['beta']
    with open(bash_filename,'w') as f:
        for dbeta in dbeta_range:
            beta=beta0+dbeta
            params1={'beta':beta}
            params1=str(params1)
            f.write(command_template.format(tensor_filename=tensor_filename_template.format(bond_dim=bond_dim,beta=beta),nLayers=nLayers,bond_dim=bond_dim,params=params1))