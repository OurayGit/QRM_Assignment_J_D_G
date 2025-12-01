#------------------ Load Libraries ------------------
library(readxl) # for reading Excel files
library(dplyr) # for data manipulation
library(lubridate) # for date handling
library(copula) # for copulas
library(ggplot2) # for plotting
library(tidyr) # for data reshaping
library(MASS) # for mvrnorm
library(patchwork)  # for arranging plots
library(scales) # for comma formatting

#--------------Load Data----------
creditportfolio <- read_excel("~/Documents/Uni/HSG/HS25/QRM/qrm25HSG_assignmentdata/qrm25HSG_creditportfolio.xlsx")
indexes <- read_excel("~/Documents/Uni/HSG/HS25/QRM/qrm25HSG_assignmentdata/qrm25HSG_indexes.xlsx", na = "#N/A N/A", skip = 1)
indexes <- indexes[, c(1, 2, 4)]  # Date, SPI, SPX
colnames(indexes) <- c("Date", "SPI", "SPX")

#------CLEAN SPI DATA---------------

spi_data <- indexes %>%
  dplyr::select(Date, SPI) %>%
  dplyr::mutate(
    Date = as.Date(Date),
    SPI  = as.numeric(SPI)
  ) %>%
  na.omit() %>%     # remove rows where SPI is NA
  arrange(Date)


#------CLEAN SPX DATA-------------

spx_data <- indexes %>%
  dplyr::select(Date, SPX) %>%
  dplyr::mutate(
    Date = as.Date(Date),
    SPX  = as.numeric(SPX)
  ) %>%
  na.omit() %>%     # remove rows where SPX is NA
  arrange(Date)


#------WEEKLY LOG-RETURNS: SPI------

spi_weekly <- spi_data %>%
  mutate(
    ISOyear = isoyear(Date),
    ISOweek = isoweek(Date)
  ) %>%
  group_by(ISOyear, ISOweek) %>%
  slice_tail(n = 1) %>%       # last trading day of each ISO week
  ungroup() %>%
  arrange(Date) %>%
  mutate(
    wr_SPI = log(SPI) - lag(log(SPI))
  )

#------WEEKLY LOG-RETURNS: SPX------

spx_weekly <- spx_data %>%
  mutate(
    ISOyear = isoyear(Date),
    ISOweek = isoweek(Date)
  ) %>%
  group_by(ISOyear, ISOweek) %>%
  slice_tail(n = 1) %>%       # last trading day of ISO week
  ungroup() %>%
  arrange(Date) %>%
  mutate(
    wr_SPX = log(SPX) - lag(log(SPX))
  )

#------ALIGN SPI & SPX WEEKLY RETURNS------

indexes_weekly_log <- spi_weekly %>%
  dplyr::select(ISOyear, ISOweek, Date_SPI = Date, wr_SPI) %>%
  inner_join(
    spx_weekly %>%
      dplyr::select(ISOyear, ISOweek, Date_SPX = Date, wr_SPX),
    by = c("ISOyear", "ISOweek")
  ) %>%
  filter(!is.na(wr_SPI), !is.na(wr_SPX)) %>%
  arrange(Date_SPI)   # consistent ordering

# Θ: matrix of weekly log-returns
Theta <- as.matrix(indexes_weekly_log[, c("wr_SPI", "wr_SPX")])

#------MLE of M2-------------

# Compute sample mean vector (MLE for μ) and compute sample covariance matrix (MLE for Σ)
mu_hat <- colMeans(Theta)
n <- nrow(Theta)
Sigma_hat <- (n-1)/n * cov(Theta)   # MLE for Σ
mu_hat 
Sigma_hat

#------MLE of M3------------

# Step 1: Marginal parameters (MLE for Gaussian)
mu_marg <- colMeans(Theta)
sigma_marg <- apply(Theta, 2, sd)  # standard deviation 

# Step 2: Transform Marginal parameters to uniforms using Gaussian CDFs
u <- pnorm(Theta[,1], mean = mu_marg[1], sd = sigma_marg[1])
v <- pnorm(Theta[,2], mean = mu_marg[2], sd = sigma_marg[2])
U <- cbind(u, v)  # data for copula fitting

#Without Package

# t-copula log-likelihood function
loglik_tcop <- function(params, U) {
  rho <- params[1]     # correlation, -1 < rho < 1
  nu  <- params[2]     # degrees of freedom, nu > 2
  
  if(rho <= -0.999 | rho >= 0.999 | nu <= 2) return(1e6)
  
  x1 <- qt(U[,1], df = nu)
  x2 <- qt(U[,2], df = nu)
  
  # univariate t densities
  f1 <- dt(x1, df = nu)
  f2 <- dt(x2, df = nu)
  
  # bivariate t density
  detR <- 1 - rho^2
  z <- (x1^2 - 2*rho*x1*x2 + x2^2) / detR
  f2b <- gamma((nu+2)/2)/ (gamma(nu/2) * nu*pi*sqrt(detR)) * (1 + z/nu)^(-(nu+2)/2)
  
  ll <- sum(log(f2b) - log(f1) - log(f2))
  return(-ll)  # negative log-likelihood for minimization
}

# starting values
rho_init <- cor(Theta[,1], Theta[,2])
nu_init  <- 8
start <- c(rho_init, nu_init)

fit <- optim(start, loglik_tcop, U = U,
             method = "L-BFGS-B",
             lower = c(-0.999, 2.01),
             upper = c(0.999, 200))
rho_hat1 <- fit$par[1]
nu_hat1  <- fit$par[2]
rho_hat1
nu_hat1

# Step 3: Fit t-copula using MLE
t_cop <- tCopula(dim=2, dispstr = "un")  # 2D t-copula, unrestricted correlation
fit <- fitCopula(t_cop, data = U, method = "ml")
theta_hat <- fit@estimate
rho_hat   <- theta_hat[1]
df_hat    <- theta_hat[2]
rho_hat
df_hat

# Create t-copula with estimated parameters
t.cop.sim <- tCopula(param = rho_hat,
                     df = df_hat,
                     dim = 2,
                     dispstr = "un")

# Simulate new points
n_replica <- 500
U_replica <- rCopula(n_replica, t.cop.sim)

# Original points (from your data)
df_original <- data.frame(U1 = U[,1], U2 = U[,2])

# Simulated points
df_sim <- data.frame(U1 = U_replica[,1], U2 = U_replica[,2])

# Plot overlay
ggplot() +
  geom_point(data = df_original, aes(x = U1, y = U2), color = "black") +  # original points
  geom_point(data = df_sim, aes(x = U1, y = U2), color = "red") +       # simulated points
  labs(title = "Original (black) and simulated (blue) points from fitted t-copula") +
  theme_minimal()

#-------Question 5 - Simulate M1------------

set.seed(1)  # for reproducibility

# Number of simulations
n_sim <- 10000

#Compute s_k
# Extract factor loadings as a matrix
a_k <- as.matrix(creditportfolio[, c("a_k1", "a_k2")])  # n_counterparties x 2

# Use Sigma_hat as the covariance matrix of factors
SigmaTheta <- Sigma_hat

# Compute idiosyncratic variance for each counterparty
s2_k <- apply(a_k, 1, function(row) t(row) %*% SigmaTheta %*% row)
s_k <- sqrt(s2_k)   # Idiosyncratic standard deviation

# Extract other parameters from dataset
lambda <- creditportfolio$lambda_k
a1 <- creditportfolio$a_k1
a2 <- creditportfolio$a_k2
pi_k <- creditportfolio$pi_k      # unconditional PD
E_k <- creditportfolio$`Exposure USD`        # exposure
R_k <- creditportfolio$R_k        # recovery rate
n_counterparties <- nrow(creditportfolio)

# --- Step 1: Simulate Theta from empirical distribution (model M1) ---
Theta_sim <- Theta[sample(1:nrow(Theta), n_sim, replace = TRUE), ]  # n_sim x 2

# --- Simulate Yk for each counterparty ---
Y_sim <- matrix(NA, nrow = n_sim, ncol = n_counterparties)

for (k in 1:n_counterparties) {
  eps <- rnorm(n_sim)  # idiosyncratic shocks
  Y_sim[,k] <- sqrt(lambda[k]) * (a1[k] * Theta_sim[,1] + a2[k] * Theta_sim[,2]) +
    sqrt(1 - lambda[k]) * s_k[k] * eps
}

# --- Derive thresholds d_k from unconditional PD ---

d_k1 <- numeric(n_counterparties)

for(k in 1:n_counterparties){
  d_k1[k] <- quantile(Y_sim[,k], probs = pi_k[k])
}
# Add thresholds to your dataset
creditportfolio$d_k1 <- d_k1

#Plot it

# Convert matrix to data frame
Y_df <- as.data.frame(Y_sim)
colnames(Y_df) <- paste0("Counterparty_", 1:n_counterparties)
Y_df$Simulation <- 1:n_sim

# Pivot to long format for ggplot
Y_long <- pivot_longer(Y_df, cols = -Simulation, 
                       names_to = "Counterparty", values_to = "Y")

ggplot(Y_long, aes(x = Y, fill = Counterparty)) +
  geom_histogram(bins = 50, alpha = 0.5, position = "identity") +
  labs(title = "Distribution of Y_k across simulations",
       x = "Y_k", y = "Frequency") +
  theme_minimal() +
  theme(legend.position = "none")  # hide legend for 100 counterparties

#-------Question 6 - Simulate M2 / M3-----------------
#------ M2 - Simulate Θ from bivariate Gaussian-------
Theta_simBG <- MASS::mvrnorm(n = n_sim, mu = mu_hat, Sigma = Sigma_hat)  # n_sim x 2

Y_simBG <- matrix(NA, nrow = n_sim, ncol = n_counterparties)
Y_simBG <- matrix(NA, nrow = n_sim, ncol = n_counterparties)
for (k in 1:n_counterparties) {
  eps <- rnorm(n_sim)
  Y_simBG[, k] <- sqrt(lambda[k]) * (a1[k] * Theta_simBG[,1] + a2[k] * Theta_simBG[,2]) +
    sqrt(1 - lambda[k]) * s_k[k] * eps
}

d_k2 <- numeric(n_counterparties)

for(k in 1:n_counterparties){
  d_k2[k] <- quantile(Y_simBG[,k], probs = pi_k[k])
}

# Add thresholds to your dataset
creditportfolio$d_k2 <- d_k2


#------ M3 - Simulate Θ from t-Copula with Gaussian margins -----

# --- Step 1: Simulate from t-copula with Gaussian marginals ---

t_cop <- tCopula(param = fit@estimate[1], df = fit@estimate[2], dim = 2, dispstr = "un")
U_sim <- rCopula(n_sim, t_cop)  # uniform [0,1]

# Convert to Gaussian marginals
Theta_simTC <- cbind(
  qnorm(U_sim[,1], mean = mu_hat[1], sd = sqrt(Sigma_hat[1,1])),
  qnorm(U_sim[,2], mean = mu_hat[2], sd = sqrt(Sigma_hat[2,2]))
)

# --- Step 3: Simulate Y_k ---
Y_simTC <- matrix(NA, nrow = n_sim, ncol = n_counterparties)
for (k in 1:n_counterparties) {
  eps <- rnorm(n_sim)
  Y_simTC[, k] <- sqrt(lambda[k]) * (a1[k] * Theta_simTC[,1] + a2[k] * Theta_simTC[,2]) +
    sqrt(1 - lambda[k]) * s_k[k] * eps
}

d_k3 <- numeric(n_counterparties)

for(k in 1:n_counterparties){
  d_k3[k] <- quantile(Y_simTC[,k], probs = pi_k[k])
}
# Add thresholds to your dataset
creditportfolio$d_k3 <- d_k3

#-------Question 7 - Portfolio loss distribution------------ 
#-----M1------
#Compute default indicators Ik
I_sim <- sweep(Y_sim, 2, d_k1, FUN = "<=") * 1  # 1 if default, 0 else
#Compute portfolio losses for each simulation
L_sim <- I_sim %*% (E_k * (1 - R_k))  # portfolio loss
#Inspect results
summary(L_sim)
hist(L_sim, breaks = 100, main = "Simulated Portfolio Loss Distribution (M1)",
     xlab = "Portfolio Loss (USD)", col = "skyblue")

#-----M2-------

#Default indicators
I_simBG <- sweep(Y_simBG, 2, d_k2, FUN = "<=") * 1
#Portfolio losses
L_simBG <- I_simBG %*% (E_k * (1 - R_k))
#Inspect results
summary(L_simBG)
hist(L_simBG, breaks = 100, main = "Simulated Portfolio Loss Distribution (M2)",
     xlab = "Portfolio Loss (USD)", col = "lightgreen")

#-----M3-----

#Default indicators
I_simTC <- sweep(Y_simTC, 2, d_k3, FUN = "<=") * 1
#Portfolio losses
L_simTC <- I_simTC %*% (E_k * (1 - R_k))
#Inspect results
summary(L_simTC)
hist(L_simTC, breaks = 100, main = "Simulated Portfolio Loss Distribution (M3)",
     xlab = "Portfolio Loss (USD)", col = "red")

# Compute summaries
summary_M1 <- summary(L_sim)
summary_M2 <- summary(L_simBG)
summary_M3 <- summary(L_simTC)

# Combine into a simple data frame
summary_table <- data.frame(
  Statistic = c("Min", "1st Qu.", "Median", "Mean", "3rd Qu.", "Max"),
  M1 = c(summary_M1),
  M2 = c(summary_M2),
  M3 = c(summary_M3)
)

summary_table

#Total Exposure
sum(creditportfolio$`Exposure USD`)

df_all <- rbind(
  data.frame(Loss = as.numeric(L_sim), Model = "M1"),
  data.frame(Loss = as.numeric(L_simBG), Model = "M2"),
  data.frame(Loss = as.numeric(L_simTC), Model = "M3")
)

p_left <- ggplot(df_all, aes(Loss, color = Model)) +
  geom_density(linewidth = .5) +
  theme_minimal() +
  labs(title = "Portfolio Loss Density (All Models)",
       x = "Portfolio Loss (USD)",
       y = "Density") +
      theme(legend.position = "none") +
  scale_color_manual(values = c("M1" = "blue", "M2" = "green", "M3" = "red"))
p_left

tail_cutoff <- 900000   # pick where the tail begins

p_right <- ggplot(df_all[df_all$Loss > tail_cutoff, ], 
                      aes(Loss, color = Model)) +
  geom_density(linewidth = .7) +
  theme_minimal() +
  labs(title = "Tail Focus",
       x = "Log Portfolio Loss (USD)",
       y = "Density") +
  scale_color_manual(values = c("M1" = "blue", "M2" = "green", "M3" = "red"))
p_right

p_left + p_right


#-------Question 8 - VaR and ES estimation------------

# Confidence level
alpha99 <- 0.99
alpha95 <- 0.95

# Function to compute VaR and ES
compute_risk_measures <- function(losses, alpha) {
  VaR <- quantile(losses, alpha)
  ES <- mean(losses[losses >= VaR])
  return(list(VaR = VaR, ES = ES))
}

# Compute risk measures for each model
risk_M1_99 <- compute_risk_measures(L_sim, alpha99)
risk_M2_99 <- compute_risk_measures(L_simBG, alpha99)
risk_M3_99 <- compute_risk_measures(L_simTC, alpha99)

risk_M1_95 <- compute_risk_measures(L_sim, alpha95)
risk_M2_95 <- compute_risk_measures(L_simBG, alpha95)
risk_M3_95 <- compute_risk_measures(L_simTC, alpha95)

# Display results
risk_results <- data.frame(
  Model = c("M1", "M2", "M3"),
  VaR_99 = c(risk_M1_99$VaR, risk_M2_99$VaR, risk_M3_99$VaR),
  ES_99  = c(risk_M1_99$ES, risk_M2_99$ES, risk_M3_99$ES),
  VaR_95 = c(risk_M1_95$VaR, risk_M2_95$VaR, risk_M3_95$VaR),
  ES_95  = c(risk_M1_95$ES, risk_M2_95$ES, risk_M3_95$ES)
)
risk_results

#---------------- Question 9: Dynamic VaR & ES (Rolling window = 100 weeks) -----------------

# Rolling window length (10 weeks)
window <- 100
alpha <- 0.99 #change based on which VaR wanted
n_sim_dyn <- 10000

Theta_full <- Theta                           # weekly log-returns (N x 2)
dates_weeks <- indexes_weekly_log$Date_SPI    # corresponding week dates
n_weeks <- nrow(Theta_full)
n_windows <- n_weeks - window

time_points <- dates_weeks[(window+1):n_weeks]

VaR_dyn <- matrix(NA, n_windows, 3)
ES_dyn  <- matrix(NA, n_windows, 3)
colnames(VaR_dyn) <- colnames(ES_dyn) <- c("M1","M2","M3")

simulate_losses <- function(Theta_sim) {
  nS <- nrow(Theta_sim)
  Y <- matrix(NA, nS, n_counterparties)
  for(k in 1:n_counterparties){
    eps <- rnorm(nS)
    Y[,k] <- sqrt(lambda[k]) * (a1[k]*Theta_sim[,1] + a2[k]*Theta_sim[,2]) +
             sqrt(1-lambda[k]) * s_k[k] * eps
  }
  # thresholds via unconditional πk
  d_loc <- sapply(1:n_counterparties, function(k) quantile(Y[,k], pi_k[k]))
  I <- sweep(Y, 2, d_loc, "<=") * 1
  L <- I %*% (E_k * (1 - R_k))
  list(L = as.numeric(L))
}

for(i in 1:n_windows){
  # Rolling 10-week window
  idx <- i:(i + window - 1)
  Theta_train <- Theta_full[idx, ]
  
  # ---------- M1: empirical ----------
  Theta_sim1 <- Theta_train[sample(1:window, n_sim_dyn, replace = TRUE), ]
  L1 <- simulate_losses(Theta_sim1)$L
  
  # ---------- M2: Gaussian ----------
  mu_hat_win <- colMeans(Theta_train)
  Sigma_hat_win <- (window-1)/window * cov(Theta_train)
  Theta_sim2 <- MASS::mvrnorm(n_sim_dyn, mu_hat_win, Sigma_hat_win)
  L2 <- simulate_losses(Theta_sim2)$L
  
  # ---------- M3: t-copula with Gaussian margins ----------
  mu_marg <- colMeans(Theta_train)
  sd_marg <- apply(Theta_train, 2, sd)
  U_train <- cbind(
    pnorm(Theta_train[,1], mu_marg[1], sd_marg[1]),
    pnorm(Theta_train[,2], mu_marg[2], sd_marg[2])
  )
  
  fit_try <- try(fitCopula(tCopula(dim=2), U_train, method="ml"), silent=TRUE)
  
  if(inherits(fit_try, "try-error")){
    rho_emp <- cor(Theta_train)[1,2]
    U_sim <- rCopula(n_sim_dyn, normalCopula(rho_emp))
  } else {
    rho_hat <- fit_try@estimate[1]
    df_hat  <- fit_try@estimate[2]
    U_sim <- rCopula(n_sim_dyn, tCopula(rho_hat, df=df_hat, dim=2))
  }
  
  Theta_sim3 <- cbind(
    qnorm(U_sim[,1], mu_hat_win[1], sqrt(Sigma_hat_win[1,1])),
    qnorm(U_sim[,2], mu_hat_win[2], sqrt(Sigma_hat_win[2,2]))
  )
  L3 <- simulate_losses(Theta_sim3)$L
  
  # ---------- Risk measures ----------
  VaR_dyn[i,1] <- quantile(L1, alpha)
  ES_dyn[i,1]  <- mean(L1[L1 >= VaR_dyn[i,1]])
  
  VaR_dyn[i,2] <- quantile(L2, alpha)
  ES_dyn[i,2]  <- mean(L2[L2 >= VaR_dyn[i,2]])
  
  VaR_dyn[i,3] <- quantile(L3, alpha)
  ES_dyn[i,3]  <- mean(L3[L3 >= VaR_dyn[i,3]])
}

# ---------- Put into data frame ----------
dyn_df <- data.frame(
  Date = time_points,
  VaR_M1 = VaR_dyn[,1],
  VaR_M2 = VaR_dyn[,2],
  VaR_M3 = VaR_dyn[,3],
  ES_M1  = ES_dyn[,1],
  ES_M2  = ES_dyn[,2],
  ES_M3  = ES_dyn[,3]
)

# ---------- Plot VaR ----------
dyn_df %>%
  pivot_longer(starts_with("VaR"), names_to = "Model", values_to = "VaR") %>%
  ggplot(aes(Date, VaR, color = Model)) +
  geom_line() +
  theme_minimal() +
  labs(title="Dynamic VaR 99% (100-week rolling window)", y="VaR (USD)")

# ---------- Plot ES ----------
dyn_df %>%
  pivot_longer(starts_with("ES"), names_to = "Model", values_to = "ES") %>%
  ggplot(aes(Date, ES, color = Model)) +
  geom_line() +
  theme_minimal() +
  labs(title="Dynamic ES 99% (100-week rolling window)", y="ES (USD)")

#-----------Additional Graphics----

ggplot(indexes_weekly_log, aes(x = Date_SPI)) +
  geom_line(aes(y = wr_SPI, color = "SPI")) +
  geom_line(aes(y = wr_SPX, color = "SPX")) +
  scale_color_manual(
    values = c("SPI" = "#1f77b4",  # blue
               "SPX" = "#ff7f0e")  # orange
  ) +
  labs(
    title = "Weekly Log-Returns of SPI and SPX",
    x = "Date",
    y = "Weekly Log-Return",
    color = "Stock Index"
  ) +
  theme_minimal()

# Function to compute VaR per rating class

rating_levels <- c("AAA", "AA", "A", "BBB", "BB", "B", "CCC")

compute_VaR_per_rating <- function(Y_sim_model, d_k_model, creditportfolio, alpha = 0.99) {
  losses_per_rating <- lapply(rating_levels, function(r) {
    idx <- which(creditportfolio$rating == r)
    
    defaults <- sweep(Y_sim_model[, idx], 2, d_k_model[idx], "<=") * 1
    portfolio_losses <- defaults %*% (creditportfolio$`Exposure USD`[idx] * (1 - creditportfolio$R_k[idx]))
    
    data.frame(
      Rating = r,
      VaR = quantile(portfolio_losses, alpha)
    )
  })
  
  VaR_df <- do.call(rbind, losses_per_rating)
  VaR_df$Rating <- factor(VaR_df$Rating, levels = rating_levels)
  return(VaR_df)
}

# Compute 99% VaR per rating for each model
VaR_M1 <- compute_VaR_per_rating(Y_sim, creditportfolio$d_k1, creditportfolio)
VaR_M1$Model <- "M1"

VaR_M2 <- compute_VaR_per_rating(Y_simBG, creditportfolio$d_k2, creditportfolio)
VaR_M2$Model <- "M2"

VaR_M3 <- compute_VaR_per_rating(Y_simTC, creditportfolio$d_k3, creditportfolio)
VaR_M3$Model <- "M3"

VaR_all <- rbind(VaR_M1, VaR_M2, VaR_M3)

p3 <- ggplot(VaR_all, aes(x = Rating, y = VaR, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "99% VaR of Portfolio Loss per Rating Class", 
       y = "VaR (USD)", x = "Rating") +
  scale_y_continuous(labels = comma) +  # makes numbers readable
  theme_minimal()

# --- Total Exposure per Rating ---

exposure_rating <- creditportfolio %>%
  group_by(rating) %>%
  summarise(Total_Exposure = sum(`Exposure USD`))
exposure_rating$rating <- factor(exposure_rating$rating, levels = rating_levels)

p4 <- ggplot(exposure_rating, aes(x = rating, y = Total_Exposure, fill = rating)) +
  geom_bar(stat = "identity") +
  labs(title = "Total Exposure per Rating Class", y = "Exposure (USD)", x = "Rating") +
  theme_minimal() +
  theme(legend.position = "none")  # remove legend


p4 + p3
# --- PD Plot ---
pd_rating <- creditportfolio %>%
  group_by(rating) %>%
  summarise(Avg_PD = mean(pi_k))

rating_levels <- c("AAA", "AA", "A", "BBB", "BB", "B", "CCC")
pd_rating$rating <- factor(pd_rating$rating, levels = rating_levels)

p1 <- ggplot(pd_rating, aes(x = rating, y = Avg_PD, fill = rating)) +
  geom_bar(stat = "identity") +
  labs(title = "Average Probability of Default per Rating Class", 
       y = "Probability of Default", x = "Rating") +
  theme_minimal() +
  theme(legend.position = "none")

# --- Threshold Plot ---
thresholds_rating <- creditportfolio %>%
  group_by(rating) %>%
  summarise(
    d_k1_avg = mean(d_k1),
    d_k2_avg = mean(d_k2),
    d_k3_avg = mean(d_k3)
  ) %>%
  pivot_longer(cols = starts_with("d_k"), names_to = "Model", values_to = "Threshold")

thresholds_rating$rating <- factor(thresholds_rating$rating, levels = rating_levels)

p2 <- ggplot(thresholds_rating, aes(x = rating, y = Threshold, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Average Default Threshold per Rating Class", 
       y = "Threshold (d_k)", x = "Rating") +
  theme_minimal()

# --- Combine side by side ---
p1 + p2



