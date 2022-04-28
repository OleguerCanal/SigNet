# Convert output files from each method to SigNet format:

experiment_id <- "exp_oversample"
list_of_methods <- c("decompTumor2Sig", "mutSignatures", "SignatureEstimationQP", "YAPSA" ,"MutationalPatterns") #,"deconstructSigs")

for(i in 1:length(list_of_methods)){
  test_guess <- read.csv(paste("../../data/", experiment_id, "/other_methods/", list_of_methods[i], "_test/inferred.exposures.csv", sep=""), row.names = 1)
  test_guess <- t(test_guess)
  test_guess <- test_guess / rowSums(test_guess)
  write.table(test_guess, paste("../../data/", experiment_id, "/other_methods/", list_of_methods[i], "_guess.csv", sep = ""), col.names = FALSE, row.names = FALSE, sep = ",") 
}
