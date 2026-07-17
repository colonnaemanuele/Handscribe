import os
import faulthandler
from handscribe.run.config import load_config
from handscribe.run.launcher import ExperimentLauncher
from handscribe.run.processor import ExperimentProcessor

faulthandler.enable()

def run_single_experiment(args_namespace):
    processor = ExperimentProcessor(args_namespace)
    processor.setup()
    processor.load()
    processor.run()
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except:
        pass

def main():
    args = load_config()    
    launcher = ExperimentLauncher(args, args.work_dir)
    param_grids = launcher.generate_parameter_grid()

    if getattr(args, 'use_slurm', False):
        launcher.run_slurm(param_grids)
    else:
        if len(param_grids) > 1:
            for i, param_namespace in enumerate(param_grids):
                print(f"\n--- Starting Experiment {i+1}/{len(param_grids)} ---")
                run_dir = os.path.join(args.work_dir, f"run_{i}")
                os.makedirs(run_dir, exist_ok=True)
                param_namespace.work_dir = run_dir                
                run_single_experiment(param_namespace)
        else:
            run_single_experiment(param_grids[0])


if __name__ == "__main__":
    main()