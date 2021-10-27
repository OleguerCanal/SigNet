import os
from posixpath import commonpath
import sys
import subprocess
import random

import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from HyperParameterOptimizer import SearchJobInstance


class ErrorfinderJobInstance(SearchJobInstance):
    def __init__(self, id):
        super().__init__(id)
        with open('job_details.txt', 'r') as file:
            self.job_details = file.read()

    def launch(self,
               batch_size,
               lr,
               num_neurons_pos,
               num_hidden_layers_pos,
               num_neurons_neg,
               num_hidden_layers_neg ,
               lagrange_missclassification,
               lagrange_pnorm,
               lagrange_smalltozero,
               pnorm_order,
               plot=False):
        self.passed_args = locals()
        self.source = "random_low"

        shell_file = self.job_details + "#$ -o signatures-net/tmp/Cluster/errorfinder_%s_%s.out"%(self.source, str(self.id)) + '\n' + '\n'
        args = "--config_file='configs/errorfinder_random_bayesian.yaml'" # Base config file
        args += " --model_id=" + str(self.id)
        args += " --batch_size=" + str(batch_size)
        args += " --lr=" + str(lr)
        args += " --num_neurons_pos=" + str(num_neurons_pos)
        args += " --num_hidden_layers_pos=" + str(num_hidden_layers_pos)
        args += " --num_neurons_neg=" + str(num_neurons_neg)
        args += " --num_hidden_layers_neg=" + str(num_hidden_layers_neg)
        args += " --lagrange_missclassification=" + str(lagrange_missclassification)
        args += " --lagrange_pnorm=" + str(lagrange_pnorm)
        args += " --lagrange_smalltozero=" + str(lagrange_smalltozero) 
        args += " --pnorm_order=" + str(pnorm_order) 
        shell_file += "cd signatures-net/src/ ; conda activate sigs_env ; python train_errorfinder.py " + args

        create_sh_command = "echo '" + shell_file + "' | ssh cserranocolome@ant-login.linux.crg.es -T 'cat  > signatures-net/tmp/errorfinder_" + self.source + "_" + str(self.id) + ".sh'" 
        create_sh_process = subprocess.Popen(create_sh_command, shell=True)
        print("Job " + str(self.id) + ": Creating .sh file...")
        create_sh_process.wait()
        if create_sh_process.returncode != 0:
            # Error creating the .sh file
            print("There was an error creating the .sh file!")
            return 1
        print("Job " + str(self.id) + ": .sh file created!")

        submit_job_command = "ssh cserranocolome@ant-login.linux.crg.es 'qsub -N errorfinder_" + self.source + "_" + str(self.id) + " signatures-net/tmp/errorfinder_" + self.source + "_" + str(self.id) + ".sh'" 
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
        command += "cat signatures-net/tmp/errorfinder_score_%s_%s.txt"%(self.source,str(self.id))
        output = float(subprocess.check_output(
            shlex.split(command)).decode('utf-8').split("\n")[0])
        return output

    def done(self):
        import shlex
        command = "ssh cserranocolome@ant-login.linux.crg.es "
        command += "ls signatures-net/tmp/"
        output = subprocess.check_output(
            shlex.split(command)).decode('utf-8').split("\n")
        is_done = "errorfinder_score_%s_%s.txt"%(self.source,str(self.id)) in output
        return is_done

    def kill(self):
        # os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
        pass

    def end(self):
        pass


if __name__ == '__main__':
    errorfinder_job_instance = ErrorfinderJobInstance(1)
    errorfinder_job_instance.launch(500, 1e-4, 300, 3, 200, 2, 7e-3, 1e4, 1.0, 5.0, plot=True)
