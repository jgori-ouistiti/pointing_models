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

fitts_diag = function(data){
  mean_data = data %>%
    group_by(id) %>%
    summarise(avgmt=mean(mt))
  
  
  p <- ggplot(data, aes(x = id, y = mt)) +
    geom_point(color = "blue", size = 3) +   # Points
    ggtitle("ggplot2 with Additional Lines") +
    xlab("X-axis") + ylab("Y-axis")
  
  p <- p +
    geom_point(data = mean_data, aes(x=id, y=avgmt), color = "red", size = 5) +
    geom_smooth(method = "lm", col = "red")
  
  model = lm(avgmt~id, data=mean_data)
  output = list("model" = model, "p" = p)
  return(output)
}



association_diag = function(data){
  tau = cor.test(data$id, data$mt, method = "kendall")
  rho = cor.test(data$id, data$mt, method = "spearman")
  r = cor.test(data$id, data$mt, method = "pearson")
  return (list(tau, rho, r))
}

# compute kendall's tau



copula_fits = function(data){
  print("======inside=======")

  clayton_fit <-tryCatch({
  cop_model = claytonCopula(dim = 2)
  m = pobs(as.matrix(cbind(data$id,data$mt)))
  fitCopula(cop_model, m, method = 'ml')},
  error=function(e){'claytonCopula'})
  
  
  gumbel_fit <-tryCatch({
  cop_model = gumbelCopula(dim = 2)
  m = pobs(as.matrix(cbind(data$id,data$mt)))
  fitCopula(cop_model, m, method = 'ml')},
  error=function(e){'gumbelCopula'})
  
  galambos_fit <-tryCatch({
  cop_model = galambosCopula()
  m = pobs(as.matrix(cbind(data$id,data$mt)))
  fitCopula(cop_model, m, method = 'ml')},
  error=function(e){'galambosCopula'})
  
  HR_fit <-tryCatch({
  cop_model = huslerReissCopula()
  m = pobs(as.matrix(cbind(data$id,data$mt)))
  fitCopula(cop_model, m, method = 'ml')},
  error=function(e){'huslerReissCopula'})
  
  tev_fit <-tryCatch({
  cop_model = tevCopula()
  m = pobs(as.matrix(cbind(data$id,data$mt)))
  fitCopula(cop_model, m, method = 'ml')},
  error=function(e){'tevCopula'})
  
  
  t_fit <-tryCatch({
  cop_model = ellipCopula ("t", dim = 2)
  m = pobs(as.matrix(cbind(data$id,data$mt)))
  fitCopula(cop_model, m, method = 'ml')},
  error=function(e){'tCopula'})
  
  rotgumbel_fit <-tryCatch({
  cop_model = rotCopula(gumbelCopula(dim = 2))
  m = pobs(as.matrix(cbind(data$id,data$mt)))
  fitCopula(cop_model, m, method = 'ml')},
  error=function(e){'rot gumbelCopula'})
  
  rotgalambos_fit <-tryCatch({
  cop_model = rotCopula(galambosCopula())
  m = pobs(as.matrix(cbind(data$id,data$mt)))
  fitCopula(cop_model, m, method = 'ml')},
  error=function(e){'rot galambosCopula'})
  
  rotHR_fit <-tryCatch({
  cop_model = rotCopula(huslerReissCopula())
  m = pobs(as.matrix(cbind(data$id,data$mt)))
  fitCopula(cop_model, m, method = 'ml')},
  error=function(e){'rot huslerReissCopula'})
  
  normal_fit <-tryCatch({
  cop_model = normalCopula()
  m = pobs(as.matrix(cbind(data$id,data$mt)))
  fitCopula(cop_model, m, method = 'ml')},
  error=function(e){'normalCopula'})
  
  
  return(list(normal_fit, clayton_fit, gumbel_fit, galambos_fit, HR_fit, tev_fit, t_fit, rotgumbel_fit, rotgalambos_fit, rotHR_fit))
  
}

path =  "../python/JGP_per_xp.csv"
df = read.csv(path)
df <- df[df$day <= 2, ] # remove day 3 which is a replication of day 1

# W
grouped <- df %>% group_by(A, W, participant)
grouped_data <- grouped %>% group_split()

loglik_df_condition <- data.frame(P = rep(NA, 12*11*15), D = rep(NA, 12*11*15), W = rep(NA, 12*11*15), ll = rep(NA, 12*11*15), copula = rep(NA, 12*11*15), params1 = rep(NA, 12*11*15), params2 = rep(NA, 12*11*15))

df_index = 1

As = c(256, 512, 1024, 1408)
Ws = c(64, 96, 128)

copnames = c('Independent', 'Normal', 'Clayton', 'Gumbel', 'Galambos', 'HR', 't-EV', 't', 'rotated Gumbel', 'rotated Galambos', 'rotated HR')

for (group in grouped_data){
  a = unique(group$A)
  w = unique(group$W)
  p = unique(group$participant)
  row_index = (which(As == a)-1)*3 + which(Ws == w)
  data= select(group, c('participant', 'IDe.2d.', "Duration"))
  colnames(data) = c('participant', 'id', 'mt')
  fits = copula_fits(data)
  col_index = 1
  loglik_df_condition[df_index,] = c(p, a, w, 0, "Independent", NA, NA)
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
    else{loglik_df[row_index, col_index] = -Inf
      fit_ll_value = -Inf
      copname = fit
      copparams = c(NA,NA)}
    if (length(copparams)==1){copparams = c(copparams, NA)}
    loglik_df_condition[df_index + col_index-1,] = c(p, a, w, fit_ll_value, copname, copparams)
    
    col_index = col_index +1
    }
  df_index = df_index + 10
}


library(formattable)
formattable(loglik_df, list(V1 = color_tile('white', 'orange'), V2 = color_tile('white', 'orange'), V3 = color_tile('white', 'orange'), V4 = color_tile('white', 'orange'), V5 = color_tile('white', 'orange'), V6 = color_tile('white', 'orange'), V7 = color_tile('white', 'orange'), V8 = color_tile('white', 'orange'), V9 = color_tile('white', 'orange'), V10 = color_tile('white', 'orange'), V11 = color_tile('white', 'orange'), V12 = color_tile('white', 'orange'), V13 = color_tile('white', 'orange')))
write.csv(loglik_df_condition, "exchange/loglik_df_jgp_D_W_P.csv")

