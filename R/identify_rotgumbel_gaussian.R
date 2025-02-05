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



# loglik = 230, theta = 2.35

### ======== JGP 

# emg mt
beta = 0.55
sigma = 0.12
lambda = 0.46

# unif ide
min_ide = 1.38
max_ide=4.5

N = 100
cop_model = rotCopula(gumbelCopula(dim = 2, param=2))
mymvd = mvdc(copula = cop_model, margins = c('unif', 'emg'), paramMargins = list(list(min=min_ide, max = max_ide), list(mu=beta, sigma=sigma, lambda=lambda)))

R_samplesize =matrix(NA, nrow = N, ncol = 5)

k = 1
for (n in c(50, 100, 200, 500, 1000)){
  for (r in 1:N){
    print(r)
  Z <- rMvdc(n, mymvd)
  # gauss fit
  cop_model = normalCopula()
  m = pobs(as.matrix(Z))
  fit = fitCopula(cop_model, m, method = 'ml')
  ll = fit@loglik
  aic_gauss = 2 - 2*ll
  
  cop_model = rotCopula(gumbelCopula(dim = 2))
  fit = fitCopula(cop_model, m, method = 'ml')
  ll = fit@loglik
  aic_rotgumbel = 2 - 2*ll
  
  R = exp((aic_rotgumbel-aic_gauss)/2)
  R_samplesize[r,k] = R
  }
  k = k+1
}
write.csv(R_samplesize, "exchange/identifiability_joint.csv", row.names = FALSE)


