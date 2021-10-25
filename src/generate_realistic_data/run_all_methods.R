
source("https://bioconductor.org/biocLite.R")
biocLite("BiocInstaller")

install.packages("BiocManager")
BiocManager::install(c("Biostrings","BSgenome","GenomeInfoDb","GenomicRanges"))

install.packages("remotes")
remotes::install_github(repo = "steverozen/SynSigGen", ref = "1.0.6-branch")

install.packages("devtools")
devtools::install_github("WuyangFF95/SynSigRun")
library(SynSigRun)
 

if(cosmic_version == "v3"){
    path_to_sigs = "../../data/data.xlsx"
    
  }else if(cosmic_version == "v2"){
    path_to_sigs = "../../data/data_v2.xlsx"
    
  }else{
    print("Error: not implemented for this version of COSMIC")
    break
}

dataset <- "test"
RundecompTumor2SigAttributeOnly(
 input.catalog = paste("../../data/", experiment_id, "/other_methods/", dataset, "_ground.truth.syn.catalog.csv", sep = ""),
 gt.sigs.file = path_to_sigs,
 paste("../../data/", experiment_id, "/other_methods/decompTumor2Sig_test", sep=""),
 seedNumber = 1,
 test.only = FALSE,
 overwrite = FALSE
)
 
RunMutationalPatternsAttributeOnly(
  input.catalog = paste("../../data/", experiment_id, "/other_methods/", dataset, "_ground.truth.syn.catalog.csv", sep = ""),
  gt.sigs.file = path_to_sigs,
  paste("../../data/", experiment_id, "/other_methods/MutationalPatterns_test", sep=""),
  seedNumber = 1,
  test.only = FALSE,
  overwrite = FALSE
)

RunmutSignaturesAttributeOnly(
  input.catalog = paste("../../data/", experiment_id, "/other_methods/", dataset, "_ground.truth.syn.catalog.csv", sep = ""),
  gt.sigs.file = path_to_sigs,
  paste("../../data/", experiment_id, "/other_methods/mutSignatures_test", sep=""),
  seedNumber = 1,
  test.only = FALSE,
  overwrite = FALSE
)

RunSignatureEstimationQPAttributeOnly(
  input.catalog = paste("../../data/", experiment_id, "/other_methods/", dataset, "_ground.truth.syn.catalog.csv", sep = ""),
  gt.sigs.file = path_to_sigs,
  paste("../../data/", experiment_id, "/other_methods/SignatureEstimationQP_test", sep=""),
  seedNumber = 1,
  test.only = FALSE,
  overwrite = FALSE
)

RunYAPSAAttributeOnly(
  input.catalog = paste("../../data/", experiment_id, "/other_methods/", dataset, "_ground.truth.syn.catalog.csv", sep = ""),
  gt.sigs.file = path_to_sigs,
  paste("../../data/", experiment_id, "/other_methods/YAPSA_test", sep=""),
  seedNumber = 1,
  signature.cutoff = NULL,
  test.only = FALSE,
  overwrite = FALSE
)

RundeconstructSigsAttributeOnly(
  input.catalog = paste("../../data/", experiment_id, "/other_methods/", dataset, "_ground.truth.syn.catalog.csv", sep = ""),
  gt.sigs.file = path_to_sigs,
  paste("../../data/", experiment_id, "/other_methods/deconstructSigs_test", sep=""),
  seedNumber = 1,
  test.only = FALSE,
  overwrite = FALSE
)
