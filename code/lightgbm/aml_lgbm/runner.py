# Copyright (c) 2020 Microsoft
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import os
import subprocess
import logging
from aml_lgbm.logger import AmlLogPipe

MPI_RANK = int(os.environ.get('OMPI_COMM_WORLD_RANK', 0))
MPI_MODE = True if os.environ.get('OMPI_COMM_WORLD_RANK') else False

class LightGBMRunner(object):

    def __init__(self, config_file: str, train_data: str,
                 validation_data: str, parameters: dict,
                 run_context, lgbm_exec_path='lightgbm'):
        self.config_file = config_file
        self.train_data = train_data
        self.validation_data = validation_data
        self.run_context = run_context
        self.param_dict = parameters
        self.exec_path = lgbm_exec_path

    def run(self):
        logpipe = AmlLogPipe(logging.INFO, self.run_context)
        with subprocess.Popen(self.command_line, stdout=logpipe, stderr=logpipe) as s:
            logpipe.close()

        if s.returncode != 0:
            raise RuntimeError(
                f'LightGBM exited with exit code: {s.returncode}')

    @property
    def command_line(self):
        """Returns list of command line arguments"""
        command_line = [self.exec_path]

        if self.config_file:
            command_line.append(f'config={self.config_file}')
        if self.train_data:
            command_line.append(f'data={self.get_file_names(self.train_data, "train")}')
        if self.validation_data:
            command_line.append(
                f'valid={self.get_file_names(self.validation_data, "valid")}')
        if self.param_dict:
            command_line += self._dict_to_param_list(self.param_dict)

        return command_line

    def get_file_names(self, path: str, prefix:str) -> str:
        return self.load_csv_by_rank(path, prefix) if MPI_MODE else self.expand_path(path)

    @staticmethod
    def load_csv_by_rank(path:str, prefix:str) -> str:
        return os.path.join(path, f'{prefix}_{MPI_RANK}.csv')

    @staticmethod
    def _dict_to_param_list(params_dict: dict) -> list:
        """Take a dict object and convert to a string of key=value seperated by spaces"""
        param_list = []
        for key, value in params_dict.items():
            # Add the parameter to the string if it is not None
            if value:
                param_list.append(f'{key}={value}')

        return param_list

    @staticmethod
    def expand_path(path: str) -> str:
        """Take a path and return a list of files if a directory, otherwise return the path. LightGBM expects a list of files in a comma seperated string."""
        # Check if it is a directory, if so return all files in the directory
        if os.path.isdir(path):
            files = [entry.path for entry in os.scandir(
                path) if entry.is_file()]
            return ','.join(files)
        # Otherwise return the file name
        else:
            return path

    @property
    def model_file(self):
        return self.param_dict['output_model']