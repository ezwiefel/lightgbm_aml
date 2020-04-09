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

from aml_lgbm import AmlLogPipe, LightGBMRunner

from azureml.core import Run

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

    if MPI_MODE:
        lgbm_params['machines'] = create_machine_list(kwargs['local_listen_port'])

    lgbm_runner = LightGBMRunner(config_file=conf_file, train_data=train_data,
                                 validation_data=valid_data, parameters=lgbm_params, run_context=run)

    lgbm_runner.run()

    if MPI_RANK == 0:
        os.makedirs('outputs', exist_ok=True)
        shutil.copy(lgbm_runner.model_file, os.path.join('./outputs', lgbm_runner.model_file))


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
                raise ValueError(
                    f"The arguments passed must be in --arg value format. Argument '{arg[0]}' appears to have no associate value.")
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


if __name__ == "__main__":
    main()
