# Convert output files from each method to SigNet format:
# 
# experiment_id <- "exp_generator"
# test_set <- "test_generator"
# list_of_methods <- c("decompTumor2Sig", "mutSignatures", "SignatureEstimationQP", "YAPSA" ,"MutationalPatterns") #,"deconstructSigs")
# list_of_methods <- c("deconstructSigs")
# for(i in 1:length(list_of_methods)){
#   test_guess <- read.csv(paste("../../data/", experiment_id, "/", test_set, "/other_methods/", list_of_methods[i], "_test/inferred.exposures.csv", sep=""), row.names = 1)
#   test_guess <- t(test_guess)
#   test_guess <- test_guess / rowSums(test_guess)
#   write.table(test_guess, paste("../../data/", experiment_id, "/", test_set, "/other_methods/", list_of_methods[i], "_guess.csv", sep = ""), col.names = FALSE, row.names = FALSE, sep = ",") 
# }





#### REAL DATA ####
list_of_methods <- c("decompTumor2Sig", "mutSignatures", "SignatureEstimationQP", "YAPSA" ,"MutationalPatterns","deconstructSigs")

for(i in 1:length(list_of_methods)){
  test_guess <- read.csv(paste("../../data/real_data/other_methods/PCAWG_", list_of_methods[i], "/inferred.exposures.csv", sep=""), row.names = 1)
  test_guess <- t(test_guess)
  test_guess <- test_guess / rowSums(test_guess)
  write.table(test_guess, paste("../../data/real_data/other_methods/PCAWG_", list_of_methods[i], "_guess.csv", sep = ""), col.names = TRUE, row.names = 1, sep = ",") 
}
