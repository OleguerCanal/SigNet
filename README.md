# SigNet

SigNet is a package to study genetic mutational processes.
Check out our [theoretical background page](documentation/theoretical_background.md) for further information on this topic.
As of now, it contains 3 solutions:

- **[SigNet Refitter](documentation/signet_refitter.md)**: Tool for signature decomposition.
- **[SigNet Generator](documentation/signet_generator.md)**: Tool for realistic mutational data generation.
- **[SigNet Detector](documentation/signet_detector.md)**: Tool for mutational vector out-of-distribution detection.


## Readme contents

You can use SigNet in 3 different ways depending on your workflow:

1. **[Python Package](##Python-Package)**
   1. Python Package Installation
   2. Python Package Usage

2. **[Command Line Interface](##Command-Line-Interface)** (CLI)
   1. CLI Installation
   2. CLI Usage

3. **[Source Code](Source-Code)**
   1. Downloading Source Code
   2. Code-Basics


## Python Package
Recommended if you want to integrate SigNet as part of your python workflow, or intending to re-train models on custom data with limited ANN architectural changes.
You can install the python package running:

```BASH
pip install signet
```

Once installed, check out [refitter_example.py](refitter_example.py) for a usage example.

**NOTE**: _It is recommended that you work on a [custom python virtualenvironment](https://virtualenv.pypa.io/en/latest/) to avoid pavkage version mismatches._

**NOTE**: _Custom model training is relatively limited if installing SigNet as a python package, please consider [downloading the source code](### Downloading Source Code)_.


## Command Line Interface

Recommended if only interested in running SigNet modules independently and **not** willing to retrain models or change the source code.<br>
**NOTE**: _This option is only tested on Debian-based Linux distributions_.

### CLI Installation

Download the [signet exectuable](TODOlink_to_executable) 

### CLI Usage

__Refitter:__

The following example shows how to use [SigNet Refitter](documentation/signet_refitter.md).


```BASH
cd <wherever/you/downloaded/the/executable/>
./signet refitter  [--input_data INPUTFILE]
                   [--normalization {None, exome, genome, PATH_TO_ABUNDANCES}] 
                   [--output OUTPUT]
                   [--figure False]
```

- `--input_data`: Path to the file containing the mutational counts. Please refer to [Mutations Input](documentation/input_output_formats.md##Mutations-Input) for further details on the input format.

- `--normalization`: As the INPUTFILE contain counts, we need to normalize them according to the abundances of each trinucleotide on the genome region we are counting the mutations.
  - Choose `None` (default): If you don't want any normalization.
  - Choose `exome`:  If the data that is being input comes from Whole Exome Sequencing. This will normalize the counts according to the trinucleotide abundances in the exome.
  - Choose `genome`: If the data comes from Whole Genome Sequencing.
  - Set a `PATH_TO_ABUNDANCES` to use a custom normalization file. Please refer to [Normalization Input](documentation/input_output_formats.md##Mutations-Input) for further details on the input format.

- `--output` Path to the folder where all the output files (weights guesses and figures) will be stored. By default, this folder will be called "OUTPUT" and will be created in the current directory. Please refer to [SigNet Refitter Output](documentation/input_output_formats.md##Signet-Refitter-Output) for further details on the output format.

- `--figure` Whether to generate output plots or not. Possible options are `true` or `false`.


__Detector:__

```BASH
cd <wherever/you/downloaded/the/executable/>
./signet detector  [--input_data INPUTFILE]
                   [--normalization {None, exome, genome, PATH_TO_ABUNDANCES}] 
                   [--output OUTPUT]
```

(Same arguments as before)

__Generator:__

```BASH
cd <wherever/you/downloaded/the/executable/>
./signet detector  [--n_datapoints INT]
                   [--normalization {None, exome, genome, PATH_TO_ABUNDANCES}] 
                   [--output OUTPUT]
```

- `--input_data`: Path to the file containing the mutational counts. Please refer to [Mutations Input](documentation/input_output_formats.md##Mutations-Input) for further details on the input format.

(Same arguments as before)

## Source Code

Is the option which gives more flexibility.
Recommended if you want to play around with the code, re-train custom models or [do contributions](documentation/).

### Downloading Source Code


```BASH
git clone git@github.com:OleguerCanal/signatures-net.git
cd signatures-net
pip install -r requirements.txt
```

`TODO link to main class documentations`

`TODO link to page explaining how to train each model`
