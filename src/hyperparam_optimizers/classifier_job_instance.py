import os
from posixpath import commonpath
import sys
import subprocess
import random

import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from HyperParameterOptimizer import SearchJobInstance


class ClassifierJobInstance(SearchJobInstance):
    def __init__(self, id):
        super().__init__(id)
        with open('job_details.txt', 'r') as file:
            self.job_details = file.read()

    def launch(self,
               batch_size,
               lr,
               num_neurons,
               num_hidden_layers,
               plot=False):
        self.passed_args = locals()

        self.file_name = "classifier"
        shell_file = self.job_details + "#$ -o signatures-net/tmp/Cluster/%s_%s.out"%(self.file_name, str(self.id)) + '\n' + '\n'
        args = "--config_file='configs/classifier_bayesian.yaml'" # Base config file
        args += " --model_id=" + str(self.id)
        args += " --batch_size=" + str(batch_size)
        args += " --lr=" + str(lr)
        args += " --num_neurons=" + str(num_neurons)
        args += " --num_hidden_layers=" + str(num_hidden_layers)
        shell_file += "cd signatures-net/src/ ; conda activate sigs_env ; python train_classifier.py " + args

        create_sh_command = "echo '" + shell_file + "' | ssh cserranocolome@ant-login.linux.crg.es -T 'cat  > signatures-net/tmp/%s_"%self.file_name + str(self.id) + ".sh'" 
        create_sh_process = subprocess.Popen(create_sh_command, shell=True)
        print("Job " + str(self.id) + ": Creating .sh file...")
        create_sh_process.wait()
        if create_sh_process.returncode != 0:
            # Error creating the .sh file
            print("There was an error creating the .sh file!")
            return 1
        print("Job " + str(self.id) + ": .sh file created!")

        submit_job_command = "ssh cserranocolome@ant-login.linux.crg.es 'qsub -N %s_"%self.file_name + str(self.id) + " signatures-net/tmp/%s_"%self.file_name + str(self.id) + ".sh'" 
        submit_job_process = subprocess.Popen(submit_job_command, shell=True)
        print("Job " + str(self.id) + ": Running qsub...")
        submit_job_process.wait()
        if submit_job_process.returncode != 0:
            # Error running qsub
            print("There was an error running qsub!")
            return 1
        print("Job " + str(self.id) + ": qsub finished without errors!")
        return 0

    def get_result(self):
        import shlex
        command = "ssh cserranocolome@ant-login.linux.crg.es "
        command += "cat signatures-net/tmp/%s_score_%s.txt"%(self.file_name, self.id)
        output = float(subprocess.check_output(
            shlex.split(command)).decode('utf-8').split("\n")[0])
        return output

    def done(self):
        import shlex
        command = "ssh cserranocolome@ant-login.linux.crg.es "
        command += "ls signatures-net/tmp/"
        output = subprocess.check_output(
            shlex.split(command)).decode('utf-8').split("\n")
        is_done = "%s_score_%s.txt"%(self.file_name, self.id) in output
        return is_done

    def kill(self):
        # os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
        pass

    def end(self):
        pass


if __name__ == '__main__':
    classifier_job_instance = ClassifierJobInstance(1)
    classifier_job_instance.launch(500, 1e-4, 300, 3, plot=True)
