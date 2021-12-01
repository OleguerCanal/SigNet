# SigNet

## General Description 

Mutations that occur during cancer development can be classified into different mutational processes. This is done using the mutations’ sequence contexts and statistical tools that decompose the observed mutation spectra into statistically independent mutational components, or "signatures". Many so-called refitting algorithms have been developed to find a linear combination of these independent signatures that can describe the data and with this we can assign mutations from a given tumor to the underlying mutational processes. However, these algorithms were developed years ago, when a set of 30 signatures was originally identified, and they have not been tested for the newest catalogs that contain more than 70 different signatures. Furthermore, estimated per-tumor signature weights are noisy, yet their errors are rarely quantified in existing algorithms. 

We have developed a method that uses Deep Neural Networks to do signature refitting. This algorithm, called SigNet, is easily adaptable to any new catalog of mutational signatures and provides prediction intervals for the contribution of each signature in a single tumor. This showcases how, using Artificial Neural Networks, we can take advantage of the correlations between the different mutational processes that occur during carcinogenesis to obtain accurate signature decompositions, even when the number of mutations in the sampled tumor is very low. For more details on how the method works please see [link to the paper].

### What is signature refitting?

Given a set of known mutational signatures, signature refitting methods aim to find a linear combination of these signatures that can reconstruct the mutational profile of a given sample. The mutational signatures have been identified using the conventional 96 mutation type classification, considering not only the mutated base (six substitution subtypes: C>A, C>G, C>T, T>A, T>C, and T>G), but also the bases immediately 5’ and 3’. The currently most used set of mutational signatures is the one provided by COSMIC, [[1]](#1). A schematic of the signature refitting process is the following:


## How to install

pip install blablabla

## How to use

SigNet can be directly used from an executable. In this case one does not need to have any version of python installed and can be called from a terminal. It can also be installed with pip and after that, the python script signet.py can be executed. 

### Command Line Interface

The command to run SigNet from the command line is the following:
```
usage:    SigNet    [--input_data INPUTFILE]
                    [--normalization {None, exome, genome, PATH_TO_ABUNDANCES}] 
                    [--output OUTPUT]
                    [--figure False]
```

The only required argument is the `--input_data` that should provide the path to the data that wants to be analyzed. This argument and the other optional ones are explained in the following sections.

`--input_data INPUTFILE`

INPUTFILE should be the path to the file that contains the data. It should be a file that contains the mutational counts for a set of tumors. Each row should be a different sample and each column should be a trinucleotide mutation type. The different mutation types should follows the conventional 96 mutation type classification that is based on the six substitution subtypes: C>A, C>G, C>T, T>A, T>C, and T>G, as well as the nucleotides immediately 5’ and 3’ to the mutation [[1]](#1). Therefore, the shape of the data should be nx96, where n is the number of samples. It should also include a header with the trinucleotide mutation type for each column (format: A[C>A]A), and the sample ID as the index for each row. 

An example containing 5 samples can be found here:

------- PUT EXAMPLE TABLE -----------------------

`--normalization None`

The INPUTFILE should contain counts, so we need to normalize them according to the abundances of each trinucleotide on the genome region we are counting the mutations. If the data that is being input comes from Whole Exome Sequencing, the "exome" option should be used. This will normalize the counts according to the trinucleotide abundances in the exome. However, if the data comes from Whole Genome Sequencing, "genome" should be used. If the user can use any other kind of normalization, they can provide their own trinucleotide abundances. For this, they should provide a path to a file containing the abundances in two columns. The first one should be the trinucleotide, and the second one should be the abundance of that trinucleotide. The default option for normalization is None which means that no specific normalization will be applied and the data will just be normally normalized by the total number of mutations in each sample.

`--output OUTPUT`

OUTPUT should be the path to the folder where all the output files (weights guesses and figures) will be stored. By default, this folder will be called "Output" and will be created in the same directory as signet's directory. 

`--figure FALSE`

Boolean variable that sets whether the plots of the resulting weights guesses should be generated or not. The default is False, so the plots will not be generated. In the case that this is set to True, all the plots will be generated in the OUTPUT directory. 

### Output

The output of the algorithm is a text file containing the weights guesses and the error bars for each signature and sample. Each row corresponds to a sample and each column corresponds to a different signature. 

Think about how will we output the weights and error bars. Only one file? 3 files? 

## How to use other signature catalogs

Explain how can the user train the neural networks and test them with any other catalog of mutational signatures. What should the training, test sets and the signatures catalog look like. Maybe explain the basics here and refer to a wiki somewhere else? Because it might be too long.


## References
<a id="1">[1]</a> 
COSMIC. Mutational Signatures (v3.2 - March 2021).
Accessed on 10th November 2021.
https://cancer.sanger.ac.uk/signatures/sbs/
