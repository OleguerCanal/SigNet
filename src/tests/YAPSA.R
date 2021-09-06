library(YAPSA)

data(sigs)
data(cutoffs)
current_cutoff_vector <- cutoffCosmicValid_abs_df[6,]
data("lymphomaNature2013_mutCat_df")

data <- read.csv("../../data/exp_0/test_realistic/test_realistic_input.csv", header = FALSE)
label <- read.csv("../../data/exp_0/test_realistic/test_realistic_label.csv", header = FALSE)
num_mut <- label[,ncol(label)]
data <- data*num_mut
data <- t(data)
rownames(data) <- rownames(lymphomaNature2013_mutCat_df)
colnames(data) <- paste("sample", 1:ncol(data), sep = "")
data <- as.data.frame(data)
data_listsList <-  LCD_complex_cutoff_combined(
    in_mutation_catalogue_df = data,
    in_cutoff_vector = current_cutoff_vector, 
    in_signatures_df = AlexCosmicValid_sig_df, 
    in_sig_ind_df = AlexCosmicValid_sigInd_df)

exposures <- data_listsList$cohort$exposures
weights <- exposures/matrix(rep(colSums(exposures),nrow(exposures)), nrow=nrow(exposures), ncol=ncol(exposures), byrow=T)

# Make fake subgroups (all belong to the same group)
groups <- cbind(rep(1,ncol(data)),colnames(data))
colnames(groups) <- c("SUBGROUP", "PID")
groups <- as.data.frame(groups)
subgroups_df <- make_subgroups_df(groups, weights)

# Plot exposures
exposures_barplot(
  in_exposures_df = weights,
  in_signatures_ind_df = data_listsList$cohort$out_sig_ind_df,
  in_subgroups_df = subgroups_df)

# Compute confidence intervals
complete_df <- variateExp(
  in_catalogue_df = data,
  in_sig_df = data_listsList$cohort$signatures,
  in_exposures_df = exposures,
  in_sigLevel = 0.025, in_delta = 0.4)

# Normalize exposures and upper and lower bounds
complete_df_normalized <- complete_df
for(i in 1:nrow(complete_df)){
  sample_id <- complete_df$sample[i]
  normalization_constant <- sum(exposures[sample_id])
  complete_df_normalized$exposure[i] <- complete_df$exposure[i]/normalization_constant
  complete_df_normalized$lower[i] <- complete_df$lower[i]/normalization_constant
  complete_df_normalized$upper[i] <- complete_df$upper[i]/normalization_constant
}

# Plot exposures with confidence intervals
plotExposuresConfidence(
  in_complete_df = complete_df_normalized, 
  in_subgroups_df = subgroups_df,
  in_sigInd_df = data_listsList$cohort$out_sig_ind_df)
