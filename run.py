import sys
import re
import os
import uuid
import subprocess
import yaml
from misc_functions import patterns_to_remove


class ParallelRun:
    """SLURM job submission handler"""
    def __init__(self, params: dict, experiment_timestamp: str, slurm_config: dict = None, slurm_args: dict = None):
        self.params = params
        self.exp_timestamp = experiment_timestamp
        self.slurm_args = slurm_args or {}
        
        # Default SLURM configuration
        default_config = {
            'slurm_command': 'sbatch',
            'slurm_script': 'slurm/launch_experiment',
            'slurm_script_first_parameter': '--config',
            'slurm_outfolder': 'out',
            'out_extension': 'out',
            'param_extension': 'yaml',
            'slurm_stderr': '-e',
            'slurm_stdout': '-o',
        }
        
        self.config = {**default_config, **(slurm_config or {})}
        
        if "." not in sys.path:
            sys.path.extend(".")

    def launch(self, only_create=False):
        """Launch SLURM job"""
        subfolder = f"{self.exp_timestamp}_{self.params['experiment']['name']}"
        
        for pattern in patterns_to_remove:
            subfolder = re.sub(pattern, '', subfolder)
        subfolder = re.sub(r'_{2,}', '_', subfolder)
        subfolder = subfolder.replace("_0_", "_")
        subfolder = subfolder.replace(" ", "_").replace("/", "_").replace("\\", "_")
        subfolder = subfolder.rstrip('_')
        
        out_folder = os.path.join(self.config['slurm_outfolder'], subfolder)
        os.makedirs(out_folder, exist_ok=True)

        run_uuid = str(uuid.uuid4())[:8]
        out_file = f"{run_uuid}.{self.config['out_extension']}"
        out_file = os.path.join(out_folder, out_file)
        param_file = f"{run_uuid}.{self.config['param_extension']}"
        param_file = os.path.join(out_folder, param_file)
        
        with open(param_file, 'w') as f:
            yaml.dump(self.params, f, default_flow_style=False)
        
        command = [
            self.config['slurm_command'],
            self.config['slurm_stdout'],
            out_file,
            self.config['slurm_stderr'],
            out_file,
            self.config['slurm_script'],
            self.config['slurm_script_first_parameter'],
            param_file,
        ]
        
        for arg_name, arg_value in self.slurm_args.items():
            if arg_name.startswith('--'):
                command.append(f"{arg_name}={arg_value}" if arg_value else arg_name)
            else:
                command.append(f"--{arg_name}={arg_value}" if arg_value else f"--{arg_name}")
        
        if only_create:
            print(f"Creating command: {' '.join(command)}")
            return command, param_file
        else:
            print(f"Launching command: {' '.join(command)}")
            subprocess.run(command, capture_output=True, text=True)
            return None, param_file