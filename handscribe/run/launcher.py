import os
import time
import subprocess
import itertools
import copy
from argparse import Namespace
from run import ParallelRun 

class ExperimentLauncher:
    def __init__(self, args, base_work_dir):
        self.args = args
        self.base_work_dir = base_work_dir

    def _is_list_of_lists(self, value):
        return isinstance(value, list) and len(value) > 0 and isinstance(value[0], list)

    def _process_config(self, config, grid_params, prefix=''):
        config_dict = vars(config) if isinstance(config, Namespace) else config
        for key, value in config_dict.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, Namespace):
                self._process_config(value, grid_params, full_key)
            elif isinstance(value, dict):
                self._process_config(Namespace(**value), grid_params, full_key)
            elif self._is_list_of_lists(value):
                if len(value) > 1:
                    grid_params[full_key] = value
                else:
                    setattr(config, key, value[0])

    def _set_nested_value(self, config, key_path, value):
        """Imposta un valore annidato 'a.b.c' su Namespace."""
        keys = key_path.split('.')
        current = config
        for k in keys[:-1]:
            attr = getattr(current, k)
            if isinstance(attr, dict):
                setattr(current, k, Namespace(**attr))
            current = getattr(current, k)
        setattr(current, keys[-1], value)

    def generate_parameter_grid(self):
        base_config = copy.deepcopy(self.args)
        grid_params = {}
        
        self._process_config(base_config, grid_params)

        if not grid_params:
            return [base_config]

        param_names = list(grid_params.keys())
        param_values = list(grid_params.values())
        combinations = list(itertools.product(*param_values))

        print(f"Generated {len(combinations)} parameter combinations.")
        
        final_configs = []
        for combination in combinations:
            run_config = copy.deepcopy(base_config)
            
            for name, val in zip(param_names, combination):
                self._set_nested_value(run_config, name, val)
                
            final_configs.append(run_config)
        
        return final_configs

    def create_experiment_config(self, param_namespace, run_id):
        config = copy.deepcopy(param_namespace)
        if hasattr(config, 'work_dir'):
            base_name = os.path.basename(self.base_work_dir.rstrip('/'))
            config.work_dir = os.path.join(os.path.dirname(self.base_work_dir.rstrip('/')), f"{base_name}_run{run_id}")
        return config

    def run_slurm(self, param_grids):
        experiment_timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        args_dict = vars(self.args)
        slurm_config = {k: v for k, v in args_dict.items() if k.startswith('slurm_') and k not in ('slurm_args', 'slurm_dry_run')}
        slurm_args = {k[11:] if k.startswith('slurm_args_') else k: v for k, v in args_dict.items() if k == 'dataset' or k.startswith('slurm_args_')}

        commands = []
        for run_id, param_namespace in enumerate(param_grids):
            exp_config = self.create_experiment_config(param_namespace, run_id)
            parallel_run = ParallelRun(vars(exp_config), experiment_timestamp, slurm_config, slurm_args)
            command, _ = parallel_run.launch(only_create=True)
            if command: commands.append(command)

        if getattr(self.args, 'slurm_dry_run', False):
            print(f"Dry Run: {len(commands)} jobs prepared but not submitted.")
            return

        print(f"Submitting {len(commands)} jobs...")
        for cmd in commands:
            subprocess.run(cmd, check=True)