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



# loglik = 230, theta = 2.35

### ======== JGP 

# emg mt
beta = 0.55
sigma = 0.12
lambda = 0.46

# unif ide
min_ide = 1.38
max_ide=4.5

# rot gumbel
theta = 2

# t-copula
rho1 = .7


# par(mfrow = c(2, 2))
pdf(file='img/rotgumbel-normal-margins.pdf')
par(mar=c(4,5,2,2))
cop_model = rotCopula(gumbelCopula(dim = 2, param=theta))
mymvd = mvdc(copula = cop_model, margins = c('norm', 'norm'), paramMargins = list(list(mean=0, sd = 1), list(mean=0, sd = 1)))
contour(mymvd, dMvdc, xlim = c(-3, 3), ylim = c(-3, 3), nlevels = 15, xlab = 'X', ylab = 'Y', cex.lab = 2, cex.axis=2)
title(expression(paste("Rotated Gumbel ", theta," = 2, normal margins")), cex.main=2)
dev.off()

pdf(file='img/rotgumbel-estimated-margins.pdf')
par(mar=c(4,5,2,2))
cop_model = rotCopula(gumbelCopula(dim = 2, param=theta))
mymvd = mvdc(copula = cop_model, margins = c('unif', 'emg'), paramMargins = list(list(min=min_ide, max = max_ide), list(mu=beta, sigma=sigma, lambda=lambda)))
contour(mymvd, dMvdc, xlim = c(1, 5), ylim = c(0, 5), nlevels = 15, xlab = 'IDe', ylab = 'MT', cex.lab = 2, cex.axis=2)
title(expression(paste("Rotated Gumbel ", theta," = 2, estimated margins")), cex.main=2)
dev.off()

pdf(file='img/t-normal-margins.pdf')
par(mar=c(4,5,2,2))
cop_model = tCopula(rho1, dim = 2, df=3.75)
mymvd = mvdc(copula = cop_model, margins = c('norm', 'norm'), paramMargins = list(list(mean=0, sd = 1), list(mean=0, sd = 1)))
contour(mymvd, dMvdc, xlim = c(-3, 3), ylim = c(-3, 3), xlab = 'X', ylab = 'Y', cex.lab = 2, cex.axis=2)
title(expression(paste("tCopula, ", rho," = .76 ", nu, " = 3.75, normal margins")), cex.main=2)
dev.off()

pdf(file='img/t-estimated-margins.pdf')
par(mar=c(4,5,2,2))
cop_model = tCopula(rho1, dim = 2, df=3.75)
mymvd = mvdc(copula = cop_model, margins = c('unif', 'emg'), paramMargins = list(list(min=min_ide, max = max_ide), list(mu=beta, sigma=sigma, lambda=lambda)))
contour(mymvd, dMvdc, xlim = c(1, 5), ylim = c(0, 5), nlevels = 15, xlab = 'IDe', ylab = 'MT', cex.lab = 2, cex.axis=2)
title(expression(paste("tCopula, ", rho," = .76 ", nu, " = 3.75, estimated margins")), cex.main=2)
dev.off()


