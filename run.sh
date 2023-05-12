bash ./ising2d_gilt_X16_scan_params.sh

python calculate_observables_run.py --input_filenames ./data/ising2d_gilt_X16_scan_params/*.pt --output_dir ./data/ising2d_gilt_X16_scan_params_observables/ --observables magnetization


python calculate_scdims_run.py --input_filenames ./data/ising2d_gilt_X16_scan_params/*.pt --output_dir ./data/ising2d_gilt_X16_scan_params_scdims/ --output_eigvecs --output_eigvecs_dir ./data/ising2d_gilt_X16_scan_params_eigvecs/