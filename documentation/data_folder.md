# Data Folder Structure

- `data/`: Contains all the data needed for SigNet
  - `data.xlsx`: [COSMIC v3.1 Signatures](https://cancer.sanger.ac.uk/signatures/downloads/)
  - `data_v2.xlsx`: [COSMIC v2.0 Signatures](https://cancer.sanger.ac.uk/signatures/downloads/)
  - `mutation_type_order.xsls`: Mapping to correctly sort mutation types
  - `final/` 
    - `real_input.csv`, `real_label.csv`: Input and labels used to benchmark all methods
    - `detector/`: Contains all datasets needed to train the Detector module
    - `finetuner/`: Contains all datasets needed to train the Finetuner module
    - `errorfinder/`: Contains all datasets needed to train the Errorfinder module
  - `real_data/`:
    - `3mer_WG_hg37.txt`: Whole genome human abundances
    - `abundancies_trinucleotides.txt`: Whole exon human abundances
    - `sigprofiler_not_norm_PCAWG.csv`: [Real weights](https://dcc.icgc.org/releases/PCAWG/mutational_signatures/Signatures_in_Samples/SP_Signatures_in_Samples) used to create the training sets (normalized)