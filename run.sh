bash ./ising2d_gilt_X16_scan_params.sh
python calculate_observables_run.py --input_filenames ./data/ising2d_gilt_X16_scan_params/*.pt --output_dir ./data/ising2d_gilt_X16_scan_params_observables/ --observables magnetization
python calculate_scdims_run.py --input_filenames ./data/ising2d_gilt_X16_scan_params/*.pt --output_dir ./data/ising2d_gilt_X16_scan_params_scdims/ --output_eigvecs --output_eigvecs_dir ./data/ising2d_gilt_X16_scan_params_eigvecs/

bash ./ising2d_X16_scan_params.sh
python calculate_observables_run.py --input_filenames ./data/ising2d_X16_scan_params/*.pt --output_dir ./data/ising2d_X16_scan_params_observables/ --observables magnetization
python calculate_scdims_run.py --input_filenames ./data/ising2d_X16_scan_params/*.pt --output_dir ./data/ising2d_X16_scan_params_scdims/ --output_eigvecs --output_eigvecs_dir ./data/ising2d_X16_scan_params_eigvecs/

bash ./ising3d_gilt_X8_scan_params.sh
python calculate_observables_run.py --input_filenames ./data/ising3d_gilt_X8_scan_params/*.pt --output_dir ./data/ising3d_gilt_X8_scan_params_observables/ --observables magnetization
python calculate_scdims_run.py --input_filenames ./data/ising3d_gilt_X8_scan_params/*.pt --output_dir ./data/ising3d_gilt_X8_scan_params_scdims/ --output_eigvecs --output_eigvecs_dir ./data/ising3d_gilt_X8_scan_params_eigvecs/

bash ./ising3d_X8_scan_params.sh
python calculate_observables_run.py --input_filenames ./data/ising3d_X8_scan_params/*.pt --output_dir ./data/ising3d_X8_scan_params_observables/ --observables magnetization
python calculate_scdims_run.py --input_filenames ./data/ising3d_X8_scan_params/*.pt --output_dir ./data/ising3d_X8_scan_params_scdims/ --output_eigvecs --output_eigvecs_dir ./data/ising3d_X8_scan_params_eigvecs/

bash ./aklt3d_X8_scan_params_line.sh
python calculate_observables_run.py --input_filenames ./data/aklt3d_X8_scan_params_line/*.pt --output_dir ./data/aklt3d_X8_scan_params_line_observables/ --observables magnetizationX magnetizationY magnetizationZ
python calculate_scdims_run.py --input_filenames ./data/aklt3d_X8_scan_params_line/*.pt --output_dir ./data/aklt3d_X8_scan_params_line_scdims/ --output_eigvecs --output_eigvecs_dir ./data/aklt3d_X8_scan_params_line_eigvecs/

bash ./aklt3d_X10_scan_params_line.sh
python calculate_observables_run.py --input_filenames ./data/aklt3d_X10_scan_params_line/*.pt --output_dir ./data/aklt3d_X10_scan_params_line_observables/ --observables magnetizationX magnetizationY magnetizationZ
python calculate_scdims_run.py --input_filenames ./data/aklt3d_X10_scan_params_line/*.pt --output_dir ./data/aklt3d_X10_scan_params_line_scdims/ --output_eigvecs --output_eigvecs_dir ./data/aklt3d_X10_scan_params_line_eigvecs/


