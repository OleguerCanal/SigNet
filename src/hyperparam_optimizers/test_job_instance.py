import os
import sys
import subprocess
import random

import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from HyperParameterOptimizer import SearchJobInstance


class TestJobInstance(SearchJobInstance):
    def __init__(self, id):
        super().__init__(id)

    def launch(self,
               batch_size,
               lr,
               num_hidden_layers,
               num_units,
               plot=False):
        self.passed_args = locals()

        # ssh into server and write random number on a file
        command = 'ssh oleguer@192.168.0.19 "'
        # command += 'sleep $[ ( $RANDOM % 10 )  + 1 ]s && '
        command += 'echo ' + str(random.random()) + ' >> test/' + str(self.id) + ';"'
        self.process = subprocess.Popen(command, shell=True)

    def get_result(self):
        import shlex
        command = "ssh oleguer@192.168.0.19 "
        command += "cat test/" + str(self.id)
        output = float(subprocess.check_output(shlex.split(command)).decode('utf-8').split("\n")[0])
        return output

    def done(self):
        import shlex
        command = "ssh oleguer@192.168.0.19 "
        command += "ls test;"
        output = subprocess.check_output(shlex.split(command)).decode('utf-8').split("\n")
        is_done = str(self.id) in output
        return is_done

    def kill(self):
        os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)

    def end(self):
        pass
    
