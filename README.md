# SigNet

SigNet is a package to study genetic mutational processes.
Check out our [theoretical background page](documentation/theoretical_background.md) for further information on this topic.
As of now, it contains 3 products:

- **[SigNet Refitter](documentation/signet_refitter.md)**: Tool for signature decomposition.
- **[SigNet Generator](documentation/signet_generator.md)**: Tool for realistic data generation.
- **[SigNet Detector](documentation/signet_detector.md)**: Tool for mutational vector out-of-distribution detection.

---

## Use it!

You can use SigNet in 3 different ways depending on your workflow:

1. **[Command Line Interface](##Command-Line-Interface)** (CLI)
   1. [CLI Installation](###CLI-Installation)
   2. [CLI Usage](###CLI-Usage)

2. **[Python Package](##Python-Package)**
   1. [Python Package Installation](###Python-Package-Installation)
   2. [Python Package Usage](###Python-Package-Usage)

3. **[Source Code](Source-Code)**
   1. [Downloading Source Code](###Downloading-Source-Code)
   2. [Code-Basics](###Code-Basics)
---

## Command Line Interface

Recommended if only interested in running SigNet modules independently and **not** willing to retrain models or change the source code.<br>
**NOTE**: _This option is only supported for Debian-based Linux distributions_.

### CLI Installation

```BASH
sudo apt update
sudo apt install signet
```
### CLI Usage

The following example shows how to use [SigNet Refitter](documentation/signet_refitter.md).


```BASH
signet refitter [--input_data INPUTFILE]
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

`TODO: SigNet Detector & SigNet Generator`

---

## Python Package
Recommended if you want to integrate SigNet as part of your python workflow, intending to re-train models but 

**NOTE**: _Custom model training is relatively limited if installing SigNet as a python package, please consider [downloading the source code]()_.

### Python Package Installation

```BASH
pip install signet
```

### Python Package Usage

The following example shows how to use [SigNet Refitter](documentation/signet_refitter.md).
Please refer to the [class documentation](#todo-documentation-page) for further details such as how to use customly-trained models or different mutational type orderings.

```python
from signet import SigNetRefitter

mutation_vecs = # numpy.array(N, 96) with N mutational vectors 

signet_refitter = SigNetRefitter()
refitter_output = signet_refitter(mutation_vec=mutation_vec)

print("Signature decompositions", refitter_output["decomposition"])
print("Error Lower bounds:", refitter_output["lower_bound"])
print("Error Upper bounds:", refitter_output["upper_bound"])
print("In-distribution:", refitter_output["detector"])
```

`TODO: SigNet Detector & SigNet Generator`

--- 
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


---