import logging

#import numpy as np
from argparse import ArgumentParser
import os
import pandas as pd

from signet import DATA
from signet.modules.signet_module import SigNet

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '--experiment_id', action='store', nargs=1, type=str, required=False, default="test_0",
        help=f"Name of this inference's results"
    )

    parser.add_argument(
        '--input_data', action='store', nargs=1, type=str, required=False, default=DATA + "/datasets/example_input.csv",
        help=f'Path to the input data to be analyzed. By default it will use PCAWG dataset'
    )

    parser.add_argument(
        '--normalization', action='store', nargs=1, type=str, required=False, default=[None],
        help=f'The kind of normalization to be applied to the data. Should be either "None" (default), "exome", "genome" or a path to a file with the oppportunities.'
    )
    
    parser.add_argument(
        '--output', action='store', nargs=1, type=str, required=False, default=["Output"],
        help=f'Path to folder where all the output files will be saved. Default: "Output".'
    )

    parser.add_argument(
        '--plot_figs', action='store', nargs=1, type=bool, required=False, default=[False],
        help=f'Boolean. Whether to compute plots for the output. Default: "False".'
    )
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()

    unique_abundances = True

    if unique_abundances == True:
        # input_file_path = "data/analysis_MC3/MC3_mut_counts.csv"
        input_file_path = "data/analysis_PCAWG/PCAWG_input.csv"
        # input_file_path = "data/case_study_Moore/mut_counts_by_sample.csv"
        # input_file_path = "data/case_study_GTeX/data_by_tissue/all_tissues_input.csv"
        opportunities = "genome"
        output_path = "data/analysis_PCAWG/new_SigNet/" 
        experiment_id = "final"
        # output_path = "data/analysis_MC3/new_SigNet/" 
        plot_figs = False

        # Read data
        input_file = pd.read_csv(input_file_path, header=0, index_col=0, sep=',')

        # Load & Run signet
        signet = SigNet(opportunities_name_or_path=opportunities)
        results = signet(mutation_dataset=input_file)
        print("Results obtained!")

        # Extract results
        w, u, l, c, _ = results.get_output()

        # Store results
        results.save(path=output_path, name=experiment_id)

        # Plot figures
        results.plot_results(save=plot_figs)
    else:
        list_of_tissues = ["AdiposeTissue","AdrenalGland","Bladder","BloodVessel","Brain","Breast","CervixUteri","Colon","Esophagus","Heart","Kidney",
                    "Liver","Lung","Muscle","Nerve","Ovary","Pancreas","Pituitary","Prostate","SalivaryGland","Skin","SmallIntestine",
                    "Spleen","Stomach","Testis","Thyroid","Uterus","Vagina"]
        output_path = "data/case_study_GTeX/new_SigNet/" 

        all_w = pd.DataFrame()
        all_u = pd.DataFrame()
        all_l = pd.DataFrame()
        all_c = pd.DataFrame()
        input_cols = pd.read_csv("data/case_study_GTeX/data_by_tissue/all_tissues_input.csv").columns
        sig_names = list(pd.read_excel(os.path.join(DATA, "data.xlsx")).columns)[1:]
        for tissue in list_of_tissues:
            opportunities = "data/case_study_GTeX/abundances/abundances_%s.csv"%tissue
            input_file = pd.read_csv("data/case_study_GTeX/data_by_tissue/%s_input.csv"%tissue, header=None, index_col=False, sep=',')
            input_file.index = [tissue]
            input_file.columns = list(input_cols)[1:]
            signet = SigNet(opportunities_name_or_path=opportunities)
            results = signet(mutation_dataset=input_file)
            print("Results obtained!")

             # Extract results
            w, u, l, c, _ = results.get_output(format='pandas')
            w.columns = sig_names + ['Unknown']
            w.index = [tissue]
            all_w = pd.concat((all_w, w))
            u.columns = sig_names
            u.index = [tissue]
            all_u = pd.concat((all_u, u))
            l.columns = sig_names
            l.index = [tissue]
            all_l = pd.concat((all_l, l))
            c.columns = ["Classification"]
            c.index = [tissue]
            all_c = pd.concat((all_c, c))

        all_w.to_csv(output_path + "weight_guesses-abundances.csv", header=True, index=True)
        all_u.to_csv(output_path + "upper_bound_guesses-abundances.csv", header=True, index=True)
        all_l.to_csv(output_path + "lower_bound_guesses-abundances.csv", header=True, index=True)
        all_c.to_csv(output_path + "classification_guesses-abundances.csv", header=True, index=True)
        
