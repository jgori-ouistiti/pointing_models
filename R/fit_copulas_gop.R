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
  # loglik = 187
  cop_model = claytonCopula(dim = 2)
  m = pobs(as.matrix(cbind(data$id,data$mt)))
  clayton_fit <- try(fitCopula(cop_model, m, method = 'ml'), silent = TRUE)
  
  
  # loglik = 184
  cop_model = gumbelCopula(dim = 2)
  m = pobs(as.matrix(cbind(data$id,data$mt)))
  gumbel_fit <- try(fitCopula(cop_model, m, method = 'ml'), silent = TRUE)
  
  # loglik = 184
  cop_model = galambosCopula()
  m = pobs(as.matrix(cbind(data$id,data$mt)))
  galambos_fit <- try(fitCopula(cop_model, m, method = 'ml'), silent = TRUE)
  
  # loglik 175
  cop_model = huslerReissCopula()
  m = pobs(as.matrix(cbind(data$id,data$mt)))
  HR_fit <- try(fitCopula(cop_model, m, method = 'ml'), silent = TRUE)
  
  # loglik 184
  cop_model = tevCopula()
  m = pobs(as.matrix(cbind(data$id,data$mt)))
  tev_fit <- try(fitCopula(cop_model, m, method = 'ml'), silent = TRUE)
  
  
  # loglik 221
  cop_model = ellipCopula ("t", dim = 2)
  m = pobs(as.matrix(cbind(data$id,data$mt)))
  t_fit <- try(fitCopula(cop_model, m, method = 'ml'), silent = TRUE)
  
  # loglik = 230, theta = 2.35
  cop_model = rotCopula(gumbelCopula(dim = 2))
  m = pobs(as.matrix(cbind(data$id,data$mt)))
  rotgumbel_fit <- try(fitCopula(cop_model, m, method = 'ml'), silent = TRUE)
  
  # loglik = 229
  cop_model = rotCopula(galambosCopula())
  m = pobs(as.matrix(cbind(data$id,data$mt)))
  rotgalambos_fit <- try(fitCopula(cop_model, m, method = 'ml'), silent = TRUE)
  
  # loglik 224
  cop_model = rotCopula(huslerReissCopula())
  m = pobs(as.matrix(cbind(data$id,data$mt)))
  rotHR_fit <- try(fitCopula(cop_model, m, method = 'ml'), silent = TRUE)
  return(list(clayton_fit, gumbel_fit, galambos_fit, HR_fit, tev_fit, t_fit, rotgumbel_fit, rotgalambos_fit, rotHR_fit))
}


loglik_df <- as.data.frame(matrix(ncol = 75, nrow = 9))
loglik_part_df = data.frame(P = rep(NA, 75), I = rep(NA, 75), ll = rep(NA, 75), copula = rep(NA, 75), params = rep(NA, 75))
files <- list.files(path="../python/exchange/gop",  full.names=TRUE, recursive=FALSE)
df_index = 1
for (x in files){
  print(x)
  data = read.csv(x)
  if (nrow(data) ==0){next}
  data = select(data, c('Participant', 'IDe', "MT"))
  colnames(data) = c('participant', 'id', 'mt')
  diag = fitts_diag(data)
  print(diag$p)
  fits = copula_fits(data)
  split_parts <- unlist(strsplit(x, "/"))
  last_part <- split_parts[length(split_parts)]
  last_terms <- unlist(strsplit(last_part, "_"))
  var1 <- as.integer(last_terms[1])
  var2 <- as.integer(gsub("\\.csv", "", last_terms[2]))
  col_index = (var1)*5 + var2 +1
  
  row_index = 1
  for (fit in fits){
    print('-------')
    print(row_index)
    print(df_index)
    
    if (class(fit)=="try-error"){
      loglik_df[row_index, col_index] = NA
      loglik_part_df[df_index,] = c(var1, var2, NA, NA, NA)
    } else{
    
    loglik_df[row_index, col_index] = fit@loglik

    if("copula" %in% names(attributes(fit@copula)))
      {copname = paste("rot",attributes(fit@copula@copula)$class[1])
      copparams = attributes(fit@copula@copula)$parameters
      } else {
      copname = attributes(fit@copula)$class[1]
      copparams = attributes(fit@copula)$parameters}
   
    loglik_part_df[df_index,] = c(var1, var2, fit@loglik, copname, copparams)
    }
    row_index = row_index + 1
    df_index = df_index +1
    
  }
}


library(formattable)
formattable(loglik_df, list(V1 = color_tile('white', 'orange'), V2 = color_tile('white', 'orange'), V3 = color_tile('white', 'orange'), V4 = color_tile('white', 'orange'), V5 = color_tile('white', 'orange'), V6 = color_tile('white', 'orange'), V7 = color_tile('white', 'orange'), V8 = color_tile('white', 'orange'), V9 = color_tile('white', 'orange'), V10 = color_tile('white', 'orange'), V11 = color_tile('white', 'orange'), V12 = color_tile('white', 'orange'), V13 = color_tile('white', 'orange')))
write.csv(loglik_df, "exchange/loglik_df_gop.csv")
write.csv(loglik_part_df, "exchange/loglik_part_gop.csv")
