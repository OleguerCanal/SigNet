# Generate test set:

library(SynSigGen)
library(ICAMS)
library(readxl)

generate_realistic_data <- def(cosmic_version, dataset, large_low, experiment_id){
  name_dataset <- paste(dataset, '_realistic_', num_mut, sep = '')

  #1. Generate synthetic dataset using PCAWG sigProfiler results:
  if(cosmic_version == "v3"){
    num_sigs = 72
    path = "PCAWG_output_v3/SBS96/Suggested_Solution/COSMIC_SBS96_Decomposed_Solution/"
    
    SBS <- read_xlsx("../../data/data.xlsx")
    SBS <- SBS[,3:ncol(SBS)]
  }else if(cosmic_version == "v2"){
    num_sigs = 30
    path = "PCAWG_output_v2/SBS96/Suggested_Solution/COSMIC_SBS96_Decomposed_Solution/"
    
    SBS <- read_xlsx("../../data/data_v2.xlsx")
    SBS <- SBS[,2:ncol(SBS)]
  }else{
    print("Error: not implemented for this version of COSMIC")
    break
  }

  E <- read.csv(path + "Activities/COSMIC_SBS96_Activities_refit.txt")  
  E <- t(E)
    
  S <- ReadCatalog(path + "Signatures/COSMIC_SBS96_Signatures.txt", catalog.type = "density.signature")

  P <- GetSynSigParamsFromExposures(E)

  if(dataset == "train"){
    if(large_low == "large"){
      number_of_samples = 250000
    }else if(large_low == "low"){
      number_of_samples = 350000
    }
  } else if (dataset == "val"){
    number_of_samples = 5000
 } else if (dataset == "test"){
    number_of_samples = 19000
  }
  synthetic.exposures <- GenerateSyntheticExposures(P, num.samples = number_of_samples)
  synthetic.spectra <- CreateAndWriteCatalog(S, synthetic.exposures, my.dir = paste("../../data/", experiment_id, "/", dataset, large_low, sep=""))     


  # 2. Adapting datasets to signatures-net:

  # LABEL:
  if(dataset == "test"){
    num_muts <- c(25, 50, 100, 250, 500, 1000, 5000, 10000, 50000, 100000)
    num_muts <- rep(num_muts, each=1900)
  }

  if (dataset == "train"){
    if(large_low == "low"){
      range_muts <- c(15, 50, 100, 250, 500, 1000, 5000, 10000)
      num_samples <- c(50000,50000,50000,50000,50000,50000,50000)
    }else if( large_low == "large"){
      range_muts <- c(1e3, 5e3, 1e4, 5e4, 1e5, 5e5)
      num_samples <- c(50000,50000,50000,50000,50000)
    }
    
    num_muts <- c()
    for(i in 1:7){
      num_muts <- c(num_muts, sample(range_muts[i]:range_muts[i+1], num_samples[i], replace = TRUE))
    }
  }

  if (dataset == "val"){
    if(large_low == "low"){
      range_muts <- c(15, 50, 100, 250, 500, 1000)
    }else if( large_low == "large"){
      range_muts <- c(1e3, 5e3, 1e4, 5e5, 1e5, 5e5)
    }
    num_samples <- c(1000, 1000, 1000, 1000, 1000)
    
    num_muts <- c()
    for(i in 1:7){
      num_muts <- c(num_muts, sample(range_muts[i]:range_muts[i+1], num_samples[i], replace = TRUE))
    }
  }

  test_label <- read.csv(paste("../../data/",experiment_id, "/", dataset, large_low, "/ground.truth.syn.exposures.csv", sep = ""), row.names = 1)
  test_label <- t(test_label)
  test_label <- test_label / rowSums(test_label)
  test_label <- cbind(test_label, num_muts)

  test_label_final <- matrix(0, ncol = num_sigs, nrow = nrow(test_label)) 
  colnames(test_label_final) <- colnames(SBS)
  test_label_final <- as.data.frame(test_label_final)
  for(i in 1:ncol(test_label)){
    test_label_final[[colnames(test_label)[i]]] <- test_label[,i]
  }

  write.table(test_label_final, paste("../../data/",experiment_id,"/", name_dataset, "_label.csv", sep = ""), col.names = FALSE, row.names = FALSE, sep = ",") 

  # INPUT:

  test_label <- test_label[,1:num_sigs]
  test_input_muts <- c()
  for(i in 1:nrow(test_label)){
    prob <- rep(0, 96)
    for(j in 1:ncol(test_label)){
      prob <- prob + test_label[i,j]*SBS[,j]
    }
    if(num_muts[i]>=100000){
      test_input_muts <- rbind(test_input_muts,prob[,1]*num_muts[i])
    } 
    else{
      test_input_muts <- rbind(test_input_muts,t(rmultinom(1, size=num_muts[i], prob[,1])))
    }
    
  }
  norm_test_input_muts <- test_input_muts/num_muts
  write.table(norm_test_input_muts, paste("../../data/",experiment_id,"/", name_dataset,  "_input.csv", sep = ""), col.names = FALSE, row.names = FALSE, sep = ",") 
}


cosmic_version <- "v3" 
experiment_id <- "exp_final_realistic"

generate_realistic_data(cosmic_version, "train", "low", experiment_id)
generate_realistic_data(cosmic_version, "train", "large", experiment_id)
generate_realistic_data(cosmic_version, "val", "low", experiment_id)
generate_realistic_data(cosmic_version, "val", "large", experiment_id)
generate_realistic_data(cosmic_version, "test", "", experiment_id)


########################################################################################################################################################################
# 3. Adapting input dataset to catalog dataset: (if we want to run all the other methods)
dataset <- "test"
catalog <- read.csv(paste("../../data/", experiment_id, "/", dataset, "/ground.truth.syn.catalog.csv", sep = ""))
test_input_muts <- t(test_input_muts)
test_input_muts <- cbind(catalog[,c(1,2)], test_input_muts)
colnames(test_input_muts) <- colnames(catalog)
write.table(test_input_muts, paste("../../data/", experiment_id, "/other_methods/", dataset, "_ground.truth.syn.catalog.csv", sep = ""), row.names = FALSE, sep = ",") 