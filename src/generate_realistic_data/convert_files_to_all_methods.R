
########################################################################################################################################################################
# 3. Adapting input dataset to catalog dataset: (if we want to run all the other methods)

dataset <- "test"
experiment_id <- "exp_final"

catalog <- read.csv(paste("PCAWG_output/", dataset, "/ground.truth.syn.catalog.csv", sep = ""))
test_input <- read.csv(paste("../../data/", experiment_id, "/", dataset, "/test_input.csv", sep = ""), header = FALSE, row.names = NULL, sep = ",") 
test_label <- read.csv(paste("../../data/", experiment_id, "/", dataset, "/test_label.csv", sep = ""), header = FALSE, row.names = NULL, sep = ",") 
test_input_muts <- sapply(test_input, '*', test_label[,ncol(test_label)])
test_input_muts <- t(test_input_muts)
test_input_muts <- cbind(catalog[,c(1,2)], test_input_muts)
colnames(test_input_muts) <- c(colnames(catalog)[c(1,2)], paste("sample_", 1:nrow(test_input), sep = ""))
if(dir.exists(paste("../../data/", experiment_id, "/", dataset, "/other_methods/", sep=""))==FALSE){
  dir.create(paste("../../data/", experiment_id, "/", dataset, "/other_methods/", sep=""))
}
write.table(test_input_muts, paste("../../data/", experiment_id, "/", dataset, "/other_methods/", dataset, "_ground.truth.syn.catalog.csv", sep = ""), row.names = FALSE, sep = ",") 


