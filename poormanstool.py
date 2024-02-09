"""submit some sruns to the cluster based on dictionary of parameters"""

import subprocess
import itertools
import copy
import json
from argparse import ArgumentParser
import os

SRUN_PREFIX = """srun  --output={out_dir}/%j.out --error={out_dir}/%j.err --time={time}:00:00 --gres=gpu:{ngpus} --partition=learnlab -J {name} """
SBATCH_PREFIX="""#!/bin/bash
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --job-name={name}
#SBATCH --time=12:00:00
#SBATCH --constraint=volta32gb
#SBATCH --gres=gpu:8
#SBATCH --mem=64G
#SBATCH --array=0

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${{nodes_array[0]}}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO
export OMP_NUM_THREADS=1
export MASTER_ADDR=$head_node_ip

EXPERIMENT_PATH=scripts/runs/$SLURM_JOB_ID
{srun_prefix} \
torchrun \
--nnodes {nodes} \
--nproc_per_node {ngpus} \
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint $head_node_ip:$RANDOM \
"""

parser = ArgumentParser()
parser.add_argument(
    "config",
    type=str,
    help="path to config file expects a python file that contains a dictionary named 'config'",
)
parser.add_argument(
    "--dry-run", "-d", action="store_true", help="only print commands without executing"
)
parser.add_argument(
    "--ngpus",
    "-g",
    type=int,
    default=None,
    help="defaults to 2 locally and 8 on srun",
)
parser.add_argument("--nodes", "-n", type=int, default=0, help="number of nodes for slurm default 0 = no slurm ")
parser.add_argument("--time", "-t", type=int, default=2, help="time in hours for slurm")


def parse_json(config):

    subsweep_keys = [k for k in config.keys() if k.startswith("SUBSWEEP")]
    if len(subsweep_keys) > 0:
        commands = []
        subsweep_grids = itertools.product(*[config[k].values() for k in subsweep_keys])
        for k in subsweep_keys:
            del config[k]
        for subsweep in subsweep_grids:
            new_grid = copy.deepcopy(config)
            for v1 in subsweep:
                for k2, v2 in v1.items():
                    new_grid[k2] = v2
            # TODO: fix something here if we want swept_params to include params from subsweep
            commands.extend(parse_json(new_grid))
        return commands

    perms = list(itertools.product(*config.values()))
    commands = []

    for p in perms:
        argstr = ""
        grid_params = {}
        swept_params = {}
        for i, k in enumerate(config.keys()):
            if p[i] is None:  # to avoid setting optional parameters
                continue
            grid_params[k] = p[i]
            if len(config[k]) > 1:
                swept_params[k] = p[i]
            if isinstance(p[i], (bool, int, float)):
                v = str(p[i])
            elif type(p[i]) is list or type(p[i]) is dict:
                v = f"'{json.dumps(p[i])}'"
            else:
                assert '"' not in p[i]
                v = f'"{p[i]}"'
            argstr += f" --{str(k)}={v}"
        commands.append((argstr, grid_params))

    return commands


def srun(script_args_suffix, params, run=False, ngpus=None, time=2, nodes=0):
    if ngpus is None:
        ngpus = 8 if nodes else 2
    out_dir = params.get("out_dir", "results")
    out_dir = out_dir[:-1] if out_dir[-1] == "/" else out_dir
    name = params.get("wandb_run_name", "stool")
    prefix = ""
    if nodes == 1:
        prefix = SRUN_PREFIX.format(out_dir=out_dir, ngpus=ngpus, time=time, name=name)
        prefix += f"torchrun --nproc_per_node {ngpus} train.py config/train_gpt.py "
        command = prefix + script_args_suffix
    elif nodes > 1:
        srun_prefix = SRUN_PREFIX.format(out_dir=out_dir, ngpus=ngpus, time=time, name=name)
        prefix = SBATCH_PREFIX.format(srun_prefix=srun_prefix, nodes=nodes, ngpus=ngpus, name=name)
        prefix +=  "train.py config/train_gpt.py "
        # write out file and sbatch it
        sbatch_file = prefix + script_args_suffix
        print("saving sbatch file...\n", sbatch_file)
        with open(".stool_sbatch.sh", "w") as f:
            f.write(sbatch_file)
        command = "sbatch .stool_sbatch.sh"
    else:
        command = f"python train.py config/train_gpt.py {script_args_suffix}"
    print("running...\n", command)
    if run:
        if nodes:
            subprocess.Popen(command, shell=True)
        else:
            os.system(command)
            print("done!")
    return command


if __name__ == "__main__":
    args = parser.parse_args()
    path_to_config = args.config
    config = {}
    exec(open(path_to_config).read())  # a bit dangerous
    runs = parse_json(config)
    for r, g in runs:
        srun(r, g, not args.dry_run, args.ngpus, args.time, args.nodes)
        if not args.nodes: # if not running on slurm, just run one command
            break
