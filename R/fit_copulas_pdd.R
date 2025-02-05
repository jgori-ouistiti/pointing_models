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

path =  "../python/exchange/pdd/all.csv"
df = read.csv(path)


loglik_df <- as.data.frame(matrix(ncol = 12, nrow = 11))
loglik_part_df = data.frame(P = rep(NA, 12), ll = rep(0, 12), copula = rep(NA, 12), params1 = rep(NA, 12), params2 = rep(NA, 12))

grouped <- df %>% group_by(participant)
grouped_data <- grouped %>% group_split()

copnames = c('Independent', 'Normal', 'Clayton', 'Gumbel', 'Galambos', 'HR', 't-EV', 't', 'rotated Gumbel', 'rotated Galambos', 'rotated HR')
df_index= 1
for (group in grouped_data){
  p = unique(group$participant)
  data = select(group, c('participant', 'IDe', "MT"))
  colnames(data) = c('participant', 'id', 'mt')
  output = fitts_diag(data)
  print(output$p)
  
  fits = copula_fits(data)
  col_index = 1
  loglik_part_df[df_index,] = c( p, 0, "Independent", NA, NA)
  df_index = df_index +1
  
  for (fit in fits){
    print('-------')
    if (class(fit)[1] == "fitCopula"){
      fit_ll_value = fit@loglik
      params = fit@estimate
      if("copula" %in% names(attributes(fit@copula)))
      {copname = paste("rot",attributes(fit@copula@copula)$class[1])
      copparams = attributes(fit@copula@copula)$parameters
      } else {
        copname = attributes(fit@copula)$class[1]
        copparams = attributes(fit@copula)$parameters}
    }
    else{
      fit_ll_value = -Inf
      copname = fit
      copparams = c(NA,NA)}
    if (length(copparams)==1){copparams = c(copparams, NA)}
    loglik_part_df[df_index + col_index-1,] = c(p, fit_ll_value, copname, copparams)
    
    col_index = col_index +1
  }
  df_index = df_index + 10
}


write.csv(loglik_part_df, "exchange/loglik_df_pdd_part.csv")
