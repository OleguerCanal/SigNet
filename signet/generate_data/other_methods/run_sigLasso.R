
library("devtools")
devtools::install_github("gersteinlab/siglasso")

library(siglasso)

data(aml_7_wgs)
my_spectrum <- context2spec(aml_7_wgs)
my_sigs <- siglasso(my_spectrum, default_sig = 'cosmic_v3_wholegenome')

data(cosmic_priors)
colnames(cosmic_priors)
my_prior = ifelse(cosmic_priors$BRCA==0, 0.1, 1) #adjust the strength to 0.1, BRCA
my_sigs <- siglasso(my_spectrum, prior = my_prior[1:30]) #the last row is "other signatures"




data <- read.csv('../../../data/exp_all/test_input.csv', header = FALSE)
num_muts <- c(25,50,100,250,500,1000,5000,10000,50000,100000)
final <- c()
for(i in 1:10){
  data_i <- t(data[(2780*(i-1)+1):(2780*i),])
  data_i <- data_i*num_muts[i]
  colnames(data_i) <- paste('sample', (2780*(i-1)+1):(2780*i), sep = '')
  rownames(data_i) <- rownames(my_spectrum)

  start.time <- Sys.time()
  my_sigs <- siglasso(data_i, default_sig = 'cosmic_v3_wholegenome')
  end.time <- Sys.time()
  time.taken <- end.time - start.time
  print(time.taken)
  
  final <- cbind(final, my_sigs)
}


finalt <- t(final)
zeros <- data.frame('SBS86'= rep(0, nrow(finalt)),'SBS87'= rep(0, nrow(finalt)),'SBS88'= rep(0, nrow(finalt)),'SBS89'= rep(0, nrow(finalt)),'SBS90'= rep(0, nrow(finalt)))
finalt <- cbind(finalt, zeros)
write.table(finalt, '../../../data/sigLasso_guess.csv', sep=',')
