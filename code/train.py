# Copyright (c) 2020 Microsoft
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import click
import os
import subprocess
from threading import Event, Thread

from azureml.core import Run

TASK_CHOICES = click.Choice(['train', 'predict', 'convert_model', 'refit'])
OBJECTIVE_CHOICES = click.Choice(['regression', 'regression_l1', 'huber', 'fair', 'poisson', 'quantile', 'mape', 'gamma',
                                  'tweedie', 'binary', 'multiclass', 'multiclassova', 'cross_entropy', 'cross_entropy_lambda', 'lambdarank', 'rank_xendcg'])
BOOSTING_CHOICE = click.Choice(['gbdt', 'rf', 'dart', 'goss'])
TREE_LEARNER = click.Choice(['serial', 'feature', 'data', 'voting'])

# Determine the MPI_RANK and MPI_MODE
MPI_RANK = os.environ.get('OMPI_COMM_WORLD_RANK', 0)
MPI_MODE = True if os.environ.get('OMPI_COMM_WORLD_RANK') else False
if MPI_MODE:
    print('OpenMPI cluster run detected:')
    print(f"    MPI Rank: {MPI_RANK}")
    print(f"    MPI World Size: {os.environ.get('OMPI_COMM_WORLD_SIZE')}")


# Instantiate the run object from AzureML. We'll use this for logging, etc.
run = Run.get_context()


# NOTE: For any hyperparameter that you would like to tune, it must included
#       as a command line argument here.

# TODO: Allow for any arbitrary argument to be passed here - and pass that to LightGBM as a command line argument.

# Click is used as an alternative to argparse. It allows the use of
# Python decorators to pass CLI arguments.
@click.command("LightGBM Parallel")
@click.option('--conf_file', type=click.Path(exists=True, dir_okay=False), default=None, required=False, help="Path of the LightGBM config file")
@click.option('--train_data_directory', type=click.Path(exists=True, file_okay=False, dir_okay=True), help='Path(s) of training data. LightGBM will train from this data')
@click.option('--valid_data_directory', type=click.Path(exists=True, file_okay=False, dir_okay=True), help='Path(s) of validation data. LightGBM will output metrics from this data')
@click.option("--task", type=TASK_CHOICES, default=None, help="The LightGBM task type")
@click.option('--objective', type=OBJECTIVE_CHOICES, default=None, help='The objective of the LightGBM training')
@click.option('--boosting', type=BOOSTING_CHOICE, default=None, help='The type of boosting to use')
@click.option('--tree_learner', type=TREE_LEARNER, help='The learner used for LightGBM. For parallel processing, use feature, data, or voting')
@click.option('--num_iterations', type=str)
@click.option('--num_threads', type=int, help="The number of threads for LightGBM. Default is the number of threads in OpenMP.")
@click.option('--num_leaves', type=int, help='The max number of leaves in one tree, Default is 31')
@click.option('--learning_rate', type=float, help='The shrinkage rate. In dart it also affects on normalization weights of dropped trees')
def main(train_data_directory, valid_data_directory, conf_file, **lgbm_params):
    """Run LightGBM in parallel"""
    # OMPI_COMM_WORLD_RANK environment variable is set by OpenMPI in parallel context.

    _run_lgbm(_prepare_lgbm_command_line(conf_file=conf_file, train_data=train_data_directory,
                                     validation_data=valid_data_directory, param_dict=lgbm_params))


def _parse_files_from_directory(directory):
    """Take a directory and return a list of files in the directory. LightGBM expects a list of files in a comma seperated string."""
    files = [entry.path for entry in os.scandir(directory) if entry.is_file()]
    return ','.join(files)



def _dict_to_param_string(params_dict: dict) -> str:
    """Take a dict object and convert to a string of key=value seperated by spaces"""
    param_string = []
    for key, value in params_dict.items():
        # Add the parameter to the string if it is not None
        if value:
            param_string.append(f'{key}={value}')

    return " ".join(param_string)


def _prepare_lgbm_command_line(conf_file: str = None, train_data: str = None, validation_data: str = None, param_dict: dict = None) -> str:
    """Prepare the string that will be passed to the shell to start the LGBM run"""
    command_list = ['lightgbm']

    if conf_file:
        command_list.append(f'config={conf_file}')
    if train_data:
        command_list.append(f'data={_parse_files_from_directory(train_data)}')
    if validation_data:
        command_list.append(
            f'valid={_parse_files_from_directory(validation_data)}')
    if param_dict:
        command_list.append(_dict_to_param_string(param_dict))

    return " ".join(command_list)


def _parse_log_file():
    pass


def _run_lgbm(command_line: str):
    subprocess.call(command_line) 


if __name__ == "__main__":
    main()
