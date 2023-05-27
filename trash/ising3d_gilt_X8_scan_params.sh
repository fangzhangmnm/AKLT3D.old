#!/bin/bash
device=${1:-cuda:0}
python HOTRG_run.py --filename data/ising3d_gilt_X8_scan_params/beta_0.2116544.pt --nLayers 60 --max_dim 8 --mcf_enabled --gilt_enabled --model Ising3D --params "{'beta': 0.2116544, 'h': 0}" --device cuda:1

python HOTRG_run.py --filename data/ising3d_gilt_X8_scan_params/beta_0.2136544.pt --nLayers 60 --max_dim 8 --mcf_enabled --gilt_enabled --model Ising3D --params "{'beta': 0.2136544, 'h': 0}" --device cuda:1

python HOTRG_run.py --filename data/ising3d_gilt_X8_scan_params/beta_0.2156544.pt --nLayers 60 --max_dim 8 --mcf_enabled --gilt_enabled --model Ising3D --params "{'beta': 0.2156544, 'h': 0}" --device cuda:1

python HOTRG_run.py --filename data/ising3d_gilt_X8_scan_params/beta_0.2176544.pt --nLayers 60 --max_dim 8 --mcf_enabled --gilt_enabled --model Ising3D --params "{'beta': 0.2176544, 'h': 0}" --device cuda:1

python HOTRG_run.py --filename data/ising3d_gilt_X8_scan_params/beta_0.2196544.pt --nLayers 60 --max_dim 8 --mcf_enabled --gilt_enabled --model Ising3D --params "{'beta': 0.2196544, 'h': 0}" --device cuda:1

python HOTRG_run.py --filename data/ising3d_gilt_X8_scan_params/beta_0.2216544.pt --nLayers 60 --max_dim 8 --mcf_enabled --gilt_enabled --model Ising3D --params "{'beta': 0.2216544, 'h': 0}" --device cuda:1

python HOTRG_run.py --filename data/ising3d_gilt_X8_scan_params/beta_0.2236544.pt --nLayers 60 --max_dim 8 --mcf_enabled --gilt_enabled --model Ising3D --params "{'beta': 0.2236544, 'h': 0}" --device cuda:1

python HOTRG_run.py --filename data/ising3d_gilt_X8_scan_params/beta_0.2256544.pt --nLayers 60 --max_dim 8 --mcf_enabled --gilt_enabled --model Ising3D --params "{'beta': 0.2256544, 'h': 0}" --device cuda:1

python HOTRG_run.py --filename data/ising3d_gilt_X8_scan_params/beta_0.2276544.pt --nLayers 60 --max_dim 8 --mcf_enabled --gilt_enabled --model Ising3D --params "{'beta': 0.2276544, 'h': 0}" --device cuda:1

python HOTRG_run.py --filename data/ising3d_gilt_X8_scan_params/beta_0.2296544.pt --nLayers 60 --max_dim 8 --mcf_enabled --gilt_enabled --model Ising3D --params "{'beta': 0.2296544, 'h': 0}" --device cuda:1

python HOTRG_run.py --filename data/ising3d_gilt_X8_scan_params/beta_0.2316544.pt --nLayers 60 --max_dim 8 --mcf_enabled --gilt_enabled --model Ising3D --params "{'beta': 0.2316544, 'h': 0}" --device cuda:1

