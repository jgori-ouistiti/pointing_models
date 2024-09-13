# Copula package
library(copula)
library(emg)

association_diag = function(data){
  tau = cor.test(data$id, data$mt, method = "kendall")
  rho = cor.test(data$id, data$mt, method = "spearman")
  r = cor.test(data$id, data$mt, method = "pearson")
  return (list(tau, rho, r))
}

 

  

data = read.csv("../python/fitts_csv_GOP.csv")
data = data[c(2,3,4,5,6)]
colnames(data) = c("participant","mt","strategy","repetition","id")

associations = association_diag(data) # tau = 0.49

cop_model = ellipCopula ("t", dim = 2)
m = pobs(as.matrix(cbind(data$id,data$mt)))
t_fit <- try(fitCopula(cop_model, m, method = 'ml'), silent=TRUE)
# rho = 0.668, df = 16.9
hist(data$id)
gamma_fit = MASS::fitdistr(data$id, 'gamma')
x = seq(1,9, length=100)
y = dgamma(x,  shape = unname(gamma_fit$estimate)[1], rate = unname(gamma_fit$estimate)[2])
plot(x,y,  xlab = "", ylab = "")


hist(data$mt)
m = emg.mle(data$mt)
# mu = 0.5293452 sigma = 0.1890695  lambda = 1/mean = 1.3338371
