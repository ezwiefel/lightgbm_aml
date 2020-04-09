# Copyright (c) 2020 Microsoft
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import click
import os
import subprocess
import re
import logging
from threading import Event, Thread
import shutil

from aml_logging_wrapper import AmlLogPipe

from azureml.core import Run

TASK_CHOICES = click.Choice(['train', 'predict', 'convert_model', 'refit'])
OBJECTIVE_CHOICES = click.Choice(['regression', 'regression_l1', 'huber', 'fair', 'poisson', 'quantile', 'mape', 'gamma',
                                  'tweedie', 'binary', 'multiclass', 'multiclassova', 'cross_entropy', 'cross_entropy_lambda', 'lambdarank', 'rank_xendcg'])
BOOSTING_CHOICE = click.Choice(['gbdt', 'rf', 'dart', 'goss'])
TREE_LEARNER = click.Choice(['serial', 'feature', 'data', 'voting'])

# Determine the MPI_RANK and MPI_MODE
# OMPI_COMM_WORLD_RANK environment variable is set by OpenMPI in parallel context.
MPI_RANK = int(os.environ.get('OMPI_COMM_WORLD_RANK', 0))
MPI_MODE = True if os.environ.get('OMPI_COMM_WORLD_RANK') else False
if MPI_MODE:
    print('OpenMPI cluster run detected:')
    print(f"    MPI Rank: {MPI_RANK}")
    print(f"    MPI World Size: {os.environ.get('OMPI_COMM_WORLD_SIZE')}")


# Instantiate the run object from AzureML. We'll use this for logging, etc.
run = Run.get_context()


# Click is used as an alternative to argparse. It allows the use of
# Python decorators to pass CLI arguments.
@click.command("LightGBM Parallel", context_settings={"ignore_unknown_options": True, 'allow_extra_args': True})
@click.option('--conf_file', type=click.Path(exists=True, dir_okay=False),
              default=None, required=False, help="Path of the LightGBM config file")
@click.option('--train_data', type=click.Path(exists=True, file_okay=True, dir_okay=True),
              help='Path(s) of training data. LightGBM will train from this data')
@click.option('--valid_data', type=click.Path(exists=True, file_okay=True, dir_okay=True),
              help='Path(s) of validation data. LightGBM will output metrics from this data')
@click.option('--output_model', type=str, default='LightGBM_model.txt',
              help="The filename where the saved model will be stored.")
@click.option('--local_listen_port', type=int, default=12400)
@click.pass_context
def main(context, train_data, valid_data, conf_file, **kwargs):
    """Run LightGBM in parallel"""

    # Parse the unknown and extra known arguments into a single list
    lgbm_params = {**kwargs, **parse_unknown_args(context.args)}

    lgbm_params['machines'] = create_machine_list(kwargs['local_listen_port'])

    command_line = prepare_lgbm_command_line(conf_file=conf_file, train_data=train_data,
                                             validation_data=valid_data, param_dict=lgbm_params)

    run_lgbm(command_line)

    if MPI_RANK == 0:
        capture_model_file(filename=lgbm_params['output_model'])


def create_machine_list(port: int) -> str:
    iplist = []
    for ip in os.environ.get('AZ_BATCH_NODE_LIST').split(';'):
        iplist.append(f'{ip}:{port}')

    return ','.join(iplist)


def parse_unknown_args(arg_list: list) -> dict:
    """Takes a list of args and creates a dict object.

    Looks for '--' starting a string and uses that for a 

    Example: 

        parse_unknown_args(['--foo', 'bar', 'baz', '--test', 1])

        returns: {"foo": "bar", "test": 1}
    """

    param_dict = {}
    for arg in pairwise(arg_list):
        if arg[0].startswith('--'):
            if arg[1].startswith('--'):
                raise ValueError(f"The arguments passed must be in --arg value format. Argument '{arg[0]}' appears to have no associate value.")
            param_dict[arg[0].strip('--')] = arg[1]

    return param_dict


def pairwise(iterable) -> zip:
    """s -> (s0,s1), (s1,s2), (s2, s3), ...

    From https://docs.python.org/3/library/itertools.html
    """
    from itertools import tee
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def expand_path(path: str) -> str:
    """Take a path and return a list of files if a directory, otherwise return the path. LightGBM expects a list of files in a comma seperated string."""
    # Check if it is a directory, if so return all files in the directory
    if os.path.isdir(path):
        files = [entry.path for entry in os.scandir(path) if entry.is_file()]
        return ','.join(files)
    # Otherwise return the file name
    else:
        return path


def dict_to_param_list(params_dict: dict) -> list:
    """Take a dict object and convert to a string of key=value seperated by spaces"""
    param_list = []
    for key, value in params_dict.items():
        # Add the parameter to the string if it is not None
        if value:
            param_list.append(f'{key}={value}')

    return param_list


def prepare_lgbm_command_line(conf_file: str = None,
                              train_data: str = None,
                              validation_data: str = None,
                              param_dict: dict = None) -> list:
    """Prepare the string that will be passed to the shell to start the LGBM run"""
    command_list = ['/usr/local/bin/lightgbm']

    if conf_file:
        command_list.append(f'config={conf_file}')
    if train_data:
        command_list.append(f'data={expand_path(train_data)}')
    if validation_data:
        command_list.append(
            f'valid={expand_path(validation_data)}')
    if param_dict:
        command_list += dict_to_param_list(param_dict)
        # if len(param_list) != 0:
        #     command_list += param_list

    return command_list


def run_lgbm(command_line: str):
    logpipe = AmlLogPipe(logging.INFO, Run.get_context())
    with subprocess.Popen(command_line, stdout=logpipe, stderr=logpipe) as s:
        logpipe.close()

    if s.returncode != 0:
        raise RuntimeError(f'Lightgbm exited with exit code: {s.returncode}')


def capture_model_file(filename: str = "LightGBM_model.txt"):
    """LightGBM will save the model as a text file in the current directory. 
       By default the name is 'LightGBM_model.txt' 

       This function moves the file to the outputs folder to be captured by AML
    """
    os.makedirs('outputs', exist_ok=True)
    shutil.copy(filename, os.path.join('./outputs', filename))


if __name__ == "__main__":
    main()
