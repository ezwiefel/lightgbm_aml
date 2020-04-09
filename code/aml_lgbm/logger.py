# Copyright (c) 2020 Microsoft
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import logging
import threading
import os
import subprocess
import re
import sys

MPI_RANK = int(os.environ.get('OMPI_COMM_WORLD_RANK', 0))
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

class AmlLogPipe(threading.Thread):

    def __init__(self, level, run_context):
        """Setup the object with a logger and a loglevel
        and start the thread
        """
        threading.Thread.__init__(self)
        self.run_context = run_context
        self.daemon = False
        self.level = level
        self.fdRead, self.fdWrite = os.pipe()
        self.pipeReader = os.fdopen(self.fdRead)
        self.start()

    def fileno(self):
        """Return the write file descriptor of the pipe
        """
        return self.fdWrite

    def process_line(self, line):
        """Parses a single line of output and logs the result to Azure Machine Learning"""

        RESULT_MATCH_STRING = r'Iteration:(?P<iteration>\d+)\,\s(?P<train_test>training|valid_1)\s(?P<metric>\w+)@?(\d?)\s:\s(?P<value>[+-]?\d+\.\d+)'
        ITERATION_MATCH_STRING = r'(\d+\.\d+)\sseconds\selapsed\,\sfinished\siteration\s(\d+)'
        DATALOAD_MATCH_STRING = r'Finished\sloading\sdata\sin\s(\d+\.\d+)\sseconds'

        result_match = re.search(RESULT_MATCH_STRING, line, re.I)
        if result_match and MPI_RANK == 0:
            r = result_match.groupdict()
            r['train_test'] = "validation" if r['train_test'] == 'valid_1' else r['train_test']
            self.run_context.log(
                f"{r['train_test']}_{r['metric']}", float(r["value"]))

        # iteration_match = re.search(ITERATION_MATCH_STRING, line, re.I)
        # if iteration_match and MPI_RANK == 0:
        #     self.run_context.log_row(name="Iteration Time", Rank=int(MPI_RANK), Iteration=int(iteration_match.group(2)),
        #                        ElapsedSec=float(iteration_match.group(1)))

        dataload_match = re.search(DATALOAD_MATCH_STRING, line, re.I)
        if dataload_match and MPI_RANK == 0:
            self.run_context.log(
                "Data Load Time", float(dataload_match.group(1)))

    def run(self):
        """Run the thread, logging everything.
        """
        for line in iter(self.pipeReader.readline, ''):
            self.process_line(line)
            logging.log(self.level, line.strip('\n'))
            print(line)

        self.pipeReader.close()

    def close(self):
        """Close the write end of the pipe.
        """
        os.close(self.fdWrite)


# For testing
if __name__ == "__main__":
    import sys
    from azureml.core import Run

    logpipe = AmlLogPipe(logging.INFO, Run.get_context())
    with subprocess.Popen(['/bin/ls'], stdout=logpipe, stderr=logpipe) as s:
        logpipe.close()

    sys.exit()
