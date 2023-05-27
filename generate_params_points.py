import numpy as np
from TNModels import AKLT3D, Ising2D, Ising3D
import itertools as it



# bash_filename_template='aklt3d_X{bond_dim}_scan_params_line.sh'
# tensor_filename_template='data/aklt3d_X{bond_dim}_scan_params_line/a1_{a1:.7f}_a2_{a2:.7f}_a3_{a3:.7f}.pt'
# command_template='python HOTRG_run.py --filename {tensor_filename} --nLayers {nLayers} --max_dim {bond_dim} --mcf_enabled --model AKLT3D --params "{params}" --device $device'
# default_params=AKLT3D.get_default_params()
# scan_param_names,scan_param_ranges,nPoints=['a1'],[[-1e-6,1e-6]],[11]
# nLayers,bond_dims=60,[10]
# append=False

bash_filename_template='ising3d_X{bond_dim}_scan_params_line.sh'
tensor_filename_template='data/ising3d_X{bond_dim}_scan_params_line/beta_{beta:.7f}.pt'
command_template='python HOTRG_run.py --filename {tensor_filename} --nLayers {nLayers} --max_dim {bond_dim} --mcf_enabled --model Ising3D --params "{params}" --device $device'
default_params=Ising3D.get_default_params()
scan_param_names,scan_param_ranges,nPoints=['beta'],[[-1e-6,1e-6]],[11] # 0.2216544
nLayers,bond_dims=60,[10]
append=True

# bash_filename_template='aklt3d_X{bond_dim}_scan_params_three.sh'
# tensor_filename_template='data/aklt3d_X{bond_dim}_scan_params_three/a1_{a1:.7f}_a2_{a2:.7f}_a3_{a3:.7f}.pt'
# command_template='python HOTRG_run.py --filename {tensor_filename} --nLayers {nLayers} --max_dim {bond_dim} --mcf_enabled --model AKLT3D --params "{params}" --device $device'
# default_params=AKLT3D.get_default_params()
# scan_param_names,scan_param_ranges,nPoints=['a1','a2','a3'],[[-1,1],[-1,1],[-1,1]],[11,11,11]
# nLayers,bond_dims=60,[10]
# append=False


for bond_dim in bond_dims:
    bash_filename=bash_filename_template.format(bond_dim=bond_dim)
    scan_param_points=[np.linspace(spr[0],spr[1],nPoint)+default_params[scan_param_name] for scan_param_name,nPoint,spr in zip(scan_param_names,nPoints,scan_param_ranges)]
    if append:
        f=open(bash_filename,'a')
    else:   
        f=open(bash_filename,'w')
        f.write('#!/bin/bash\n')
        f.write('device=${1:-cuda:0}\n')

    for scan_param_values in it.product(*scan_param_points):
        params={**default_params,**dict(zip(scan_param_names,scan_param_values))}
        tensor_filename=tensor_filename_template.format(bond_dim=bond_dim,**params)
        f.write(command_template.format(tensor_filename=tensor_filename,nLayers=nLayers,bond_dim=bond_dim,params=str(params)))
        f.write('\n')

    f.close()
    print('wrote',bash_filename)