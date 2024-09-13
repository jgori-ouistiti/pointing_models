# Copula package
library(copula)


set.seed(999)


library(dplyr)


 
loglik_part_df = data.frame(P = rep(NA, 75), S = rep(NA, 75), rho1 = rep(NA, 75), nu = rep(NA, 75))
files <- list.files(path="../python/exchange/gop",  full.names=TRUE, recursive=FALSE)
df_index = 1
for (x in files){
  print(x)
  data = read.csv(x)
  if (nrow(data) ==0){next}
  data = select(data, c('Participant', 'IDe', "MT"))
  colnames(data) = c('participant', 'id', 'mt')

  cop_model = ellipCopula ("t", dim = 2)
  m = pobs(as.matrix(cbind(data$id,data$mt)))
  t_fit <- try(fitCopula(cop_model, m, method = 'ml'), silent = TRUE)
  
  split_parts <- unlist(strsplit(x, "/"))
  last_part <- split_parts[length(split_parts)]
  last_terms <- unlist(strsplit(last_part, "_"))
  var1 <- as.integer(last_terms[1])
  var2 <- as.integer(gsub("\\.csv", "", last_terms[2]))
  col_index = (var1)*5 + var2 +1
  
  row_index = 1
    if (class(t_fit)=="try-error"){
      loglik_part_df[df_index,] = c(var1, var2, NA, NA)
    } else{
      copparams = attributes(t_fit@copula)$parameters
   
    loglik_part_df[df_index,] = c(var1, var2, copparams[1],copparams[2])
    }
    df_index = df_index +1
}


write.csv(loglik_part_df, "exchange/tcop_part_gop.csv")
