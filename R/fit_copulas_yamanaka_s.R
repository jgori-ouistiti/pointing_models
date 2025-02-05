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

loglik_part_df = data.frame(P = rep(NA, 36), ll = rep(NA, 36), S =rep(NA,36), copula = rep(NA, 36), params1 = rep(NA, 36), params2 = rep(NA, 36))
file <- list.files(path="../python/exchange/yamanaka",  full.names=TRUE, recursive=FALSE)
full_data = read.csv(file)
df_index = 1

groups <- full_data %>%
  group_by(participant, strategy) %>%
  group_keys()

for (i in 1:nrow(groups)){
  P = groups$participant[i]
  S = groups$strategy[i]

  group_data <- full_data %>% 
    filter(participant == P, 
           strategy == S)
  
  if (nrow(group_data) ==0){next}
  data = select(group_data, c('participant', 'IDe', "MT"))
  colnames(data) = c('participant', 'id', 'mt')
  diag = fitts_diag(data)
  print(diag$p)
  fits = copula_fits(data)
  loglik_part_df[df_index,] = c(P, 0, S, "Independent", NA, NA)
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
    loglik_part_df[df_index,] = c(P, fit_ll_value, S, copname, copparams)
    
    fitfitrow_index = row_index + 1
    df_index = df_index +1
    row_index  = row_index +1
    
  }
}


write.csv(loglik_part_df, "exchange/loglik_yamanaka_s.csv")
