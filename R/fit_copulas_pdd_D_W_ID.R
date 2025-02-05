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

library(dplyr)
set.seed(999)


library(dplyr)
library(ggplot2)

source("utils.R")

path =  "../python/exchange/pdd/all.csv"
df = read.csv(path)




# W
grouped <- df %>% group_by(W)
grouped_data <- grouped %>% group_split()

loglik_df_condition <- data.frame(ID = rep(NA, 8*12), W = rep(NA, 8*12), ll = rep(NA, 8*12), copula = rep(NA, 8*12), params1 = rep(NA, 8*12), params2 = rep(NA, 8*12))

df_index = 1

Ws = unique(df['W'])

copnames = c('Independent', 'Normal', 'Clayton', 'Gumbel', 'Galambos', 'HR', 't-EV', 't', 'rotated Gumbel', 'rotated Galambos', 'rotated HR')

for (group in grouped_data){
  w = unique(group$W)
  id = unique(group$ID)
  row_index = which(Ws == w)
  data= select(group, c('participant', 'IDe', "MT"))
  colnames(data) = c('participant', 'id', 'mt')
  
  output = fitts_diag(data)
  print(output$p)
  
  fits = copula_fits(data)
  col_index = 1
  loglik_df_condition[df_index,] = c( id, w, 0, "Independent", NA, NA)
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
    loglik_df_condition[df_index + col_index-1,] = c(id, w, fit_ll_value, copname, copparams)
    
    col_index = col_index +1
    }
  df_index = df_index + 10
}


write.csv(loglik_df_condition, "exchange/loglik_df_pdd_D_W.csv")

