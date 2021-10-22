
########################################################################################################################################################################
# 3. Adapting input dataset to catalog dataset: (if we want to run all the other methods)

dataset <- "test"

catalog <- read.csv(paste("PCAWG_output/", dataset, "/ground.truth.syn.catalog.csv", sep = ""))
test_input_muts <- read.csv(paste("PCAWG_output/", name_dataset,  "_input_not_normalized.csv", sep = ""), col.names = FALSE, row.names = FALSE, sep = ",") 
test_input_muts <- t(test_input_muts)
test_input_muts <- cbind(catalog[,c(1,2)], test_input_muts)
colnames(test_input_muts) <- colnames(catalog)
if(dir.exists("PCAWG_output/other_methods/")==FALSE){
  dir.create("PCAWG_output/other_methods/")
}
write.table(test_input_muts, paste("PCAWG_output/other_methods/", dataset, "_ground.truth.syn.catalog.csv", sep = ""), row.names = FALSE, sep = ",") 
