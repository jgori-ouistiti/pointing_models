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