from SigProfilerExtractor import decomposition as decomp
from SigProfilerMatrixGenerator import install as genInstall
from SigProfilerExtractor import sigpro as sig
from SigProfilerExtractor import estimate_best_solution as ebs

# Reference Genome
genInstall.install('GRCh37', bash=True)

# Read data
# data = "21BRCA/21BRCA/21BRCA.txt"     # Path of your data (tab delimited file)
data = "../../data/generate_realistic_data/WG_real_data/PCAWG_sigprofiler.txt"
# SigProfiler Extractor
sig.sigProfilerExtractor("matrix", "PCAWG_output_v2", data, opportunity_genome="GRCh37", minimum_signatures=10, maximum_signatures=30, nmf_replicates=10, cosmic_version = 2)
# ebs.estimate_solution()     # Estimate best solution

# Decomposition of De Novo Signatures into COSMIC Signatures
# signatures = "/example_output/SBS96/Suggested_Solution/SBS96_De-Novo_Solution/Signatures/SBS96_De-Novo_Signatures.txt"
# activities="/example_output/SBS96/Suggested_Solution/SBS96_De-Novo_Solution/Activities/SBS96_De-Novo_Activities_refit.txt"
# samples="path/to/Samples.txt"
# output="name or path/to/output"
# decomp.decompose(signatures, activities, samples, output, genome_build="GRCh37", verbose=False)