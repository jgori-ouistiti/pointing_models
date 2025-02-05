# Copula package
library(copula)
# Fancy 3D plain scatterplots
library(scatterplot3d)
# ggplot2
library(ggplot2)
# Useful package to set ggplot plots one next to the other
library(grid)
# vinecopula
library(VineCopula)

set.seed(999)


library(dplyr)
library(ggplot2)

source("utils.R")

# strategies 0 fastest to 4 slowest

loglik_part_df = data.frame(P = rep(NA, 15), ll = rep(NA, 15), copula = rep(NA, 15), params1 = rep(NA, 15), params2 = rep(NA, 15))
files <- list.files(path="../python/exchange/gop_part",  full.names=TRUE, recursive=FALSE)
df_index = 1
for (x in files){
  print(x)
  data = read.csv(x)
  if (nrow(data) ==0){next}
  data = select(data, c('Participant', 'IDe', "MT"))
  colnames(data) = c('participant', 'id', 'mt')
  diag = fitts_diag(data)
  print(diag$p)
  fits = copula_fits(data)
  split_parts <- unlist(strsplit(x, "/"))
  last_part <- split_parts[length(split_parts)]
  last_terms <- unlist(strsplit(last_part, "_"))
  var2 <- as.integer(gsub("\\.csv", "", last_terms))
  loglik_part_df[df_index,] = c(var2, 0, "Independent", NA, NA)
  df_index = df_index +1
  row_index = 1
  for (fit in fits){
    print('-------')
    print(row_index)
    print(df_index)
    
    if (class(fit)[1] == "fitCopula"){
      fit_ll_value = fit@loglik
      params = fit@estimate
      if("copula" %in% names(attributes(fit@copula)))
      {copname = paste("rot",attributes(fit@copula@copula)$class[1])
      copparams = attributes(fit@copula@copula)$parameters
      } else {
        copname = attributes(fit@copula)$class[1]
        copparams = attributes(fit@copula)$parameters}
    }else{
    fit_ll_value = -Inf
    copname = fit
    copparams = c(NA,NA)}
    
    if (length(copparams)==1){copparams = c(copparams, NA)}
    loglik_part_df[df_index,] = c(var2, fit_ll_value, copname, copparams)
    
    fitfitrow_index = row_index + 1
    df_index = df_index +1
    row_index  = row_index +1
    
  }
}


write.csv(loglik_part_df, "exchange/loglik_go_part.csv")
