import os
import sys
import subprocess
import random
import shlex
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from HyperParameterOptimizer import SearchJobInstance


class TestJobInstance(SearchJobInstance):
    def __init__(self, id):
        super().__init__(id)

    def launch(self,
               param=123,
               plot=False):
        self.passed_args = locals()

        # ssh into server and write random number on a file
        command = 'ssh -p 1969 oleguer@62.57.177.212 "'
        command += 'python a.py"'
        print("sending", command)
        self.process = subprocess.Popen(command, shell=True)
        self.process.wait()
        print(self.process)
        print('returncode:', self.process.returncode)

    def get_result(self):
        command = "ssh  -p 1969 oleguer@62.57.177.212 "
        command += "cat test/" + str(self.id)
        output = float(subprocess.check_output(shlex.split(command)).decode('utf-8').split("\n")[0])
        return output

    def done(self):
        print(self.process.returncode)
        if self.process.returncode is not None:
            return False
        command = "ssh  -p 1969 oleguer@62.57.177.212 "
        command += "ls test;"
        output = subprocess.check_output(shlex.split(command)).decode('utf-8').split("\n")
        is_done = str(self.id) in output
        return is_done

    def kill(self):
        self

    def end(self):
        pass
    
if __name__ == "__main__":
    test_job_instance = TestJobInstance(0)
    test_job_instance.launch()
    time.sleep(1)
    print("Done: ", test_job_instance.done())
    time.sleep(4)
    print("Done: ", test_job_instance.done())
    time.sleep(4)
    print("Done: ", test_job_instance.done())
