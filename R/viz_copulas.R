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

# par(mfrow = c(2, 2))

# Gaussian, rho = .7
pdf(file='img/gaussian-normal-margins.pdf')
par(mar=c(4,5,2,2))
cop_model = normalCopula(dim = 2, param=.7)
mymvd = mvdc(copula = cop_model, margins = c('norm', 'norm'), paramMargins = list(list(mean=0, sd = 1), list(mean=0, sd = 1)))
contour(mymvd, dMvdc, xlim = c(-3, 3), ylim = c(-3, 3), nlevels = 15, xlab = 'X', ylab = 'Y', cex.lab = 2, cex.axis=2)
title(expression(paste("Gaussian ", rho," = .7, normal margins")), cex.main=2)
dev.off()

pdf(file='img/gaussian-estimated-margins.pdf')
par(mar=c(4,5,2,2))
mymvd = mvdc(copula = cop_model, margins = c('unif', 'emg'), paramMargins = list(list(min=min_ide, max = max_ide), list(mu=beta, sigma=sigma, lambda=lambda)))
contour(mymvd, dMvdc, xlim = c(1, 5), ylim = c(0, 5), nlevels = 15, xlab = 'IDe', ylab = 'MT', cex.lab = 2, cex.axis=2)
title(expression(paste("Gaussian ", rho," = .7, estimated margins")), cex.main=2)
dev.off()

# Gumbel, param = 2
pdf(file='img/gumbel-normal-margins.pdf')
par(mar=c(4,5,2,2))
cop_model = gumbelCopula(dim = 2, param=2)
mymvd = mvdc(copula = cop_model, margins = c('norm', 'norm'), paramMargins = list(list(mean=0, sd = 1), list(mean=0, sd = 1)))
contour(mymvd, dMvdc, xlim = c(-3, 3), ylim = c(-3, 3), nlevels = 15, xlab = 'X', ylab = 'Y', cex.lab = 2, cex.axis=2)
title(expression(paste("Gumbel ", theta," = 2, normal margins")), cex.main=2)
dev.off()

pdf(file='img/gumbel-estimated-margins.pdf')
par(mar=c(4,5,2,2))
mymvd = mvdc(copula = cop_model, margins = c('unif', 'emg'), paramMargins = list(list(min=min_ide, max = max_ide), list(mu=beta, sigma=sigma, lambda=lambda)))
contour(mymvd, dMvdc, xlim = c(1, 5), ylim = c(0, 5), nlevels = 15, xlab = 'IDe', ylab = 'MT', cex.lab = 2, cex.axis=2)
title(expression(paste("Gumbel ", theta," = 2, estimated margins")), cex.main=2)
dev.off()

# Clayton theta = 2.5
pdf(file='img/clayton-normal-margins.pdf')
par(mar=c(4,5,2,2))
cop_model = claytonCopula(dim = 2, param=2.5)
mymvd = mvdc(copula = cop_model, margins = c('norm', 'norm'), paramMargins = list(list(mean=0, sd = 1), list(mean=0, sd = 1)))
contour(mymvd, dMvdc, xlim = c(-3, 3), ylim = c(-3, 3), nlevels = 15, xlab = 'X', ylab = 'Y', cex.lab = 2, cex.axis=2)
title(expression(paste("Clayton ", theta," = 2.5, normal margins")), cex.main=2)
dev.off()

pdf(file='img/clayton-estimated-margins.pdf')
par(mar=c(4,5,2,2))
mymvd = mvdc(copula = cop_model, margins = c('unif', 'emg'), paramMargins = list(list(min=min_ide, max = max_ide), list(mu=beta, sigma=sigma, lambda=lambda)))
contour(mymvd, dMvdc, xlim = c(1, 5), ylim = c(0, 5), nlevels = 15, xlab = 'IDe', ylab = 'MT', cex.lab = 2, cex.axis=2)
title(expression(paste("Clayton ", theta," = 2.5, estimated margins")), cex.main=2)
dev.off()

# Galambos theta = 1.2
pdf(file='img/galambos-normal-margins.pdf')
par(mar=c(4,5,2,2))
cop_model = galambosCopula(param=1.2)
mymvd = mvdc(copula = cop_model, margins = c('norm', 'norm'), paramMargins = list(list(mean=0, sd = 1), list(mean=0, sd = 1)))
contour(mymvd, dMvdc, xlim = c(-3, 3), ylim = c(-3, 3), nlevels = 15, xlab = 'X', ylab = 'Y', cex.lab = 2, cex.axis=2)
title(expression(paste("Galambos ", theta," = 1.2, normal margins")), cex.main=2)
dev.off()

pdf(file='img/galambos-estimated-margins.pdf')
par(mar=c(4,5,2,2))
mymvd = mvdc(copula = cop_model, margins = c('unif', 'emg'), paramMargins = list(list(min=min_ide, max = max_ide), list(mu=beta, sigma=sigma, lambda=lambda)))
contour(mymvd, dMvdc, xlim = c(1, 5), ylim = c(0, 5), nlevels = 15, xlab = 'IDe', ylab = 'MT', cex.lab = 2, cex.axis=2)
title(expression(paste("Galambos ", theta," = 1.2, estimated margins")), cex.main=2)
dev.off()

# HR theta = 1.5
pdf(file='img/HR-normal-margins.pdf')
par(mar=c(4,5,2,2))
cop_model = huslerReissCopula(param=1.5)
mymvd = mvdc(copula = cop_model, margins = c('norm', 'norm'), paramMargins = list(list(mean=0, sd = 1), list(mean=0, sd = 1)))
contour(mymvd, dMvdc, xlim = c(-3, 3), ylim = c(-3, 3), nlevels = 15, xlab = 'X', ylab = 'Y', cex.lab = 2, cex.axis=2)
title(expression(paste("HR ", theta," = 1.5, normal margins")), cex.main=2)
dev.off()

pdf(file='img/HR-estimated-margins.pdf')
par(mar=c(4,5,2,2))
mymvd = mvdc(copula = cop_model, margins = c('unif', 'emg'), paramMargins = list(list(min=min_ide, max = max_ide), list(mu=beta, sigma=sigma, lambda=lambda)))
contour(mymvd, dMvdc, xlim = c(1, 5), ylim = c(0, 5), nlevels = 15, xlab = 'IDe', ylab = 'MT', cex.lab = 2, cex.axis=2)
title(expression(paste("HR ", theta," = 1.5, estimated margins")), cex.main=2)
dev.off()

# tev rho = 0.8, nu = 2
pdf(file='img/tev-normal-margins.pdf')
par(mar=c(4,5,2,2))
cop_model = tevCopula(0.8, df=2)
mymvd = mvdc(copula = cop_model, margins = c('norm', 'norm'), paramMargins = list(list(mean=0, sd = 1), list(mean=0, sd = 1)))
contour(mymvd, dMvdc, xlim = c(-3, 3), ylim = c(-3, 3), nlevels = 15, xlab = 'X', ylab = 'Y', cex.lab = 2, cex.axis=2)
title(expression(paste("tev ", rho," = .8,", nu, " = 2, normal margins")), cex.main=2)
dev.off()

pdf(file='img/tev-estimated-margins.pdf')
par(mar=c(4,5,2,2))
mymvd = mvdc(copula = cop_model, margins = c('unif', 'emg'), paramMargins = list(list(min=min_ide, max = max_ide), list(mu=beta, sigma=sigma, lambda=lambda)))
contour(mymvd, dMvdc, xlim = c(1, 5), ylim = c(0, 5), nlevels = 15, xlab = 'IDe', ylab = 'MT', cex.lab = 2, cex.axis=2)
title(expression(paste("tev ", rho," = .8,", nu, " = 2, estimated margins")), cex.main=2)
dev.off()



#  t rho = 0.7, nu = 4
pdf(file='img/t-normal-margins.pdf')
par(mar=c(4,5,2,2))
cop_model = tCopula(.7, dim = 2, df=4)
mymvd = mvdc(copula = cop_model, margins = c('norm', 'norm'), paramMargins = list(list(mean=0, sd = 1), list(mean=0, sd = 1)))
contour(mymvd, dMvdc, xlim = c(-3, 3), ylim = c(-3, 3), xlab = 'X', ylab = 'Y', cex.lab = 2, cex.axis=2)
title(expression(paste("tCopula, ", rho," = .7 ", nu, " = 4, normal margins")), cex.main=2)
dev.off()

pdf(file='img/t-estimated-margins.pdf')
par(mar=c(4,5,2,2))
mymvd = mvdc(copula = cop_model, margins = c('unif', 'emg'), paramMargins = list(list(min=min_ide, max = max_ide), list(mu=beta, sigma=sigma, lambda=lambda)))
contour(mymvd, dMvdc, xlim = c(1, 5), ylim = c(0, 5), nlevels = 15, xlab = 'IDe', ylab = 'MT', cex.lab = 2, cex.axis=2)
title(expression(paste("tCopula, ", rho," = .7 ", nu, " = 4, estimated margins")), cex.main=2)
dev.off()

# Rotated Gumbel theta = 2
pdf(file='img/rotgumbel-normal-margins.pdf')
par(mar=c(4,5,2,2))
cop_model = rotCopula(gumbelCopula(dim = 2, param=2))
mymvd = mvdc(copula = cop_model, margins = c('norm', 'norm'), paramMargins = list(list(mean=0, sd = 1), list(mean=0, sd = 1)))
contour(mymvd, dMvdc, xlim = c(-3, 3), ylim = c(-3, 3), nlevels = 15, xlab = 'X', ylab = 'Y', cex.lab = 2, cex.axis=2)
title(expression(paste("Rotated Gumbel ", theta," = 2, normal margins")), cex.main=2)
dev.off()

pdf(file='img/rotgumbel-estimated-margins.pdf')
par(mar=c(4,5,2,2))
mymvd = mvdc(copula = cop_model, margins = c('unif', 'emg'), paramMargins = list(list(min=min_ide, max = max_ide), list(mu=beta, sigma=sigma, lambda=lambda)))
contour(mymvd, dMvdc, xlim = c(1, 5), ylim = c(0, 5), nlevels = 15, xlab = 'IDe', ylab = 'MT', cex.lab = 2, cex.axis=2)
title(expression(paste("Rotated Gumbel ", theta," = 2, estimated margins")), cex.main=2)
dev.off()

# Rotated Galambos theta = 1.5
pdf(file='img/rotgalambos-normal-margins.pdf')
par(mar=c(4,5,2,2))
cop_model = rotCopula(galambosCopula(param=1.5))
mymvd = mvdc(copula = cop_model, margins = c('norm', 'norm'), paramMargins = list(list(mean=0, sd = 1), list(mean=0, sd = 1)))
contour(mymvd, dMvdc, xlim = c(-3, 3), ylim = c(-3, 3), nlevels = 15, xlab = 'X', ylab = 'Y', cex.lab = 2, cex.axis=2)
title(expression(paste("Rotated Galambos ", theta," = 1.5, normal margins")), cex.main=2)
dev.off()

pdf(file='img/rotgalambos-estimated-margins.pdf')
par(mar=c(4,5,2,2))
mymvd = mvdc(copula = cop_model, margins = c('unif', 'emg'), paramMargins = list(list(min=min_ide, max = max_ide), list(mu=beta, sigma=sigma, lambda=lambda)))
contour(mymvd, dMvdc, xlim = c(1, 5), ylim = c(0, 5), nlevels = 15, xlab = 'IDe', ylab = 'MT', cex.lab = 2, cex.axis=2)
title(expression(paste("Rotated Galambos ", theta," = 1.5, estimated margins")), cex.main=2)
dev.off()

# Rotated HR theta = 2
pdf(file='img/rotHR-normal-margins.pdf')
par(mar=c(4,5,2,2))
cop_model = rotCopula(huslerReissCopula(param=2))
mymvd = mvdc(copula = cop_model, margins = c('norm', 'norm'), paramMargins = list(list(mean=0, sd = 1), list(mean=0, sd = 1)))
contour(mymvd, dMvdc, xlim = c(-3, 3), ylim = c(-3, 3), nlevels = 15, xlab = 'X', ylab = 'Y', cex.lab = 2, cex.axis=2)
title(expression(paste("Rotated HR ", theta," = 2, normal margins")), cex.main=2)
dev.off()

pdf(file='img/rotHR-estimated-margins.pdf')
par(mar=c(4,5,2,2))
mymvd = mvdc(copula = cop_model, margins = c('unif', 'emg'), paramMargins = list(list(min=min_ide, max = max_ide), list(mu=beta, sigma=sigma, lambda=lambda)))
contour(mymvd, dMvdc, xlim = c(1, 5), ylim = c(0, 5), nlevels = 15, xlab = 'IDe', ylab = 'MT', cex.lab = 2, cex.axis=2)
title(expression(paste("Rotated HR ", theta," = 2, estimated margins")), cex.main=2)
dev.off()


# Rotated Galambos theta = .5
pdf(file='img/galambos-normal-margins_theta_5.pdf')
par(mar=c(4,5,2,2))
cop_model = galambosCopula(param=0.5)
mymvd = mvdc(copula = cop_model, margins = c('norm', 'norm'), paramMargins = list(list(mean=0, sd = 1), list(mean=0, sd = 1)))
contour(mymvd, dMvdc, xlim = c(-3, 3), ylim = c(-3, 3), nlevels = 15, xlab = 'X', ylab = 'Y', cex.lab = 2, cex.axis=2)
title(expression(paste("Galambos ", theta," = 0.5, normal margins")), cex.main=2)
dev.off()

pdf(file='img/galambos-estimated-margins_theta_5.pdf')
par(mar=c(4,5,2,2))
mymvd = mvdc(copula = cop_model, margins = c('unif', 'emg'), paramMargins = list(list(min=min_ide, max = max_ide), list(mu=beta, sigma=sigma, lambda=lambda)))
contour(mymvd, dMvdc, xlim = c(1, 5), ylim = c(0, 5), nlevels = 15, xlab = 'IDe', ylab = 'MT', cex.lab = 2, cex.axis=2)
title(expression(paste("Galambos ", theta," = .5, estimated margins")), cex.main=2)
dev.off()

# indep copula
pdf(file='img/indep-normal-margins.pdf')
par(mar=c(4,5,2,2))
cop_model = indepCopula()
mymvd = mvdc(copula = cop_model, margins = c('norm', 'norm'), paramMargins = list(list(mean=0, sd = 1), list(mean=0, sd = 1)))
contour(mymvd, dMvdc, xlim = c(-3, 3), ylim = c(-3, 3), nlevels = 15, xlab = 'X', ylab = 'Y', cex.lab = 2, cex.axis=2)
title(expression(paste("Independent, normal margins")), cex.main=2)
dev.off()

pdf(file='img/indep-estimated-margins.pdf')
par(mar=c(4,5,2,2))
mymvd = mvdc(copula = cop_model, margins = c('unif', 'emg'), paramMargins = list(list(min=min_ide, max = max_ide), list(mu=beta, sigma=sigma, lambda=lambda)))
contour(mymvd, dMvdc, xlim = c(1, 5), ylim = c(0, 5), nlevels = 15, xlab = 'IDe', ylab = 'MT', cex.lab = 2, cex.axis=2)
title(expression(paste("Independent, estimated margins")), cex.main=2)
dev.off()

