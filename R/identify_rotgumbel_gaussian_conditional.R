# Copula package
library(copula)
# Fancy 3D plain scatterplots
library(scatterplot3d)
# ggplot2
library(ggplot2)
# Useful package to set ggplot plots one next to the other
library(grid)
# vinecopula
#library(VineCopula)



library(dplyr)
library(ggplot2)
library(emg)

copula_fits_rotg_g = function(data){
 
  rotgumbel_fit <-tryCatch({
    cop_model = rotCopula(gumbelCopula(dim = 2))
    m = pobs(as.matrix(cbind(data$id,data$mt)))
    fitCopula(cop_model, m, method = 'ml')},
    error=function(e){'rot gumbelCopula'})
  
  normal_fit <-tryCatch({
    cop_model = normalCopula()
    m = pobs(as.matrix(cbind(data$id,data$mt)))
    fitCopula(cop_model, m, method = 'ml')},
    error=function(e){'normalCopula'})
  return(list(normal_fit, rotgumbel_fit))
}


files <- list.files(path="../python/exchange/jgp",  full.names=TRUE, recursive=FALSE)
len_f = length(files)
N = 100
R_samplesize =matrix(NA, nrow = N, ncol = 5)
k=1
for (n in c(50, 100, 200, 500, 1000)){
  for (r in 1:N){
    print(r)
    nf = floor(runif(1)*60) +1
    data = read.csv(files[[nf]])
    data = select(data, c('participant', 'IDe.2d.', "Duration"))
    colnames(data) = c('participant', 'id', 'mt')
    len_d = nrow(data)
    indexes = floor(runif(n)*(len_d))+1
    bootstrapped_data = data[indexes,]
    fits = copula_fits_rotg_g(bootstrapped_data)
    
    normal_fit = fits[[1]]
    rotgumbel_fit = fits[[2]]
    
    ll = normal_fit@loglik
    aic_gauss = 2 - 2*ll
    
    ll = rotgumbel_fit@loglik
    aic_rotgumbel = 2 - 2*ll
    
    R = exp((aic_rotgumbel-aic_gauss)/2)
    R_samplesize[r,k] = R
    
  }
  k = k+1
  
  }

write.csv(R_samplesize, "exchange/identifiability_bootstrap.csv", row.names = FALSE)



# BELOW does not work because we can't use Ccopula on rotated copulas

# ### 
# 
# # emg mt
# beta = 0.55
# sigma = 0.12
# lambda = 0.46
# 
# # unif ide
# min_ide = 1.38
# max_ide=4.5
# 
# N = 100
# cop_model = rotCopula(gumbelCopula(dim = 2, param=2))
# mymvd = mvdc(copula = cop_model, margins = c('unif', 'emg'), paramMargins = list(list(min=min_ide, max = max_ide), list(mu=beta, sigma=sigma, lambda=lambda)))
# source("gen_t_copula.R")
# 
# 
# 
# R_samplesize =matrix(NA, nrow = N, ncol = 4)
# 
# k = 1
# for (n in c(50, 100, 200, 500)){
#   for (r in 1:N){
#     print(r)
#     blocks = n/25
#     block_levels = runif(blocks)
#     blockrMvdc(block_levels, 25, mymvd)
#   # gauss fit
#   cop_model = normalCopula()
#   m = pobs(as.matrix(Z))
#   fit = fitCopula(cop_model, m, method = 'ml')
#   ll = fit@loglik
#   aic_gauss = 2 - 2*ll
#   
#   cop_model = rotCopula(gumbelCopula(dim = 2))
#   fit = fitCopula(cop_model, m, method = 'ml')
#   ll = fit@loglik
#   aic_rotgumbel = 2 - 2*ll
#   
#   R = exp((aic_rotgumbel-aic_gauss)/2)
#   R_samplesize[r,k] = R
#   }
#   k = k+1
# }
# write.csv(R_samplesize, "exchange/identifiability_joint.csv", row.names = FALSE)


