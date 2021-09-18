# load libs and data
library(randtoolbox)
load("restless_data.RData")  # loads the data from the restless bandit as a df called "rb_data"

# define globals
dat <- rb_data
id_list <- unique(dat$id)  # extract IDs for the loop
final_data <- matrix(NA, nrow = length(id_list), ncol = 4)  # initiate empty matrix for the results
startIter <- 100
fullIter <- 5

# define kalman filter
kalman_filter <- function(choice, reward, noption, mu0, sigma0_sq, sigma_xi_sq, sigma_epsilon_sq) {
  nt <- length(choice)
  no <- noption
  m <- matrix(mu0, ncol = no, nrow = nt + 1)
  v <- matrix(sigma0_sq, ncol = no, nrow = nt + 1)
  for (t in 1:nt) {
    kt <- rep(0, no)
    kt[choice[t]] <- (v[t, choice[t]] + sigma_xi_sq)/(v[t, choice[t]] + sigma_xi_sq +
                                                        sigma_epsilon_sq)
    m[t + 1, ] <- m[t, ] + kt * (reward[t] - m[t, ])
    v[t + 1, ] <- (1 - kt) * (v[t, ] + sigma_xi_sq)
  }
  return(list(m = m, v = v))
}

# define starting values generation function
generate_starting_values <- function(n, min, max) {
  if (length(min) != length(max))
    stop("min and max should have the same length")
  dim <- length(min)
  # generate Sobol values
  start <- sobol(n, dim = dim)
  # transform these to lie between min and max on each dimension
  for (i in 1:ncol(start)) {
    start[, i] <- min[i] + (max[i] - min[i]) * start[, i]
  }
  return(start)
}

# define UCB functions
ucb_choice_prob <- function(m, ppsd, gamma, beta) {
  prob <- exp(gamma * (m + beta * ppsd))
  prob <- prob/rowSums(prob)
  return(prob)
}

kf_ucb_negLogLik_t2 <- function(par, data) {
  gamma <- exp(par[1])
  beta <- exp(par[2])
  mu0 <- par[3]
  sigma0_sq <- exp(par[4])
  sigma_xi_sq <- exp(par[5])
  sigma_epsilon_sq <- 16
  choice <- data$arm  # first change
  reward <- data$payoff
  kf <- kalman_filter(choice, reward, 6, mu0, sigma0_sq, sigma_xi_sq, sigma_epsilon_sq)  # second change
  m <- kf$m
  ppsd <- sqrt(kf$v + sigma_xi_sq)
  p <- ucb_choice_prob(m, ppsd, gamma, beta)
  lik <- p[cbind(1:nrow(data), choice)]
  negLogLik <- -sum(log(lik))
  if (is.na(negLogLik) | negLogLik == Inf)
    negLogLik <- 1e+300
  return(negLogLik)
}

# loop through all participants (enjoy waiting)
for (i in 1:length(id_list)) {
  tdat <- subset(dat, id == id_list[i])
  starting_values <- generate_starting_values(600, min = c(log(0.001), log(0.001),
                                                           -10, log(1e-04), log(1e-04)), max = c(log(10), log(10), 10, log(10000), log(100)))
  opt <- apply(starting_values, 1, function(x) optim(x, fn = kf_ucb_negLogLik_t2,
                                                     data = tdat, control = list(maxit = startIter)))
  starting_values_2 <- lapply(opt[order(unlist(lapply(opt, function(x) x$value)))[1:5]],
                              function(x) x$par)
  opt <- lapply(starting_values_2, optim, fn = kf_ucb_negLogLik_t2, data = tdat)
  bestopt <- opt[[which.min(unlist(lapply(opt, function(x) x$value)))]]
  # save iteration ID, gamma and convergence variable to matrix
  final_data[i, 1] <- id_list[i]
  final_data[i, 2] <- bestopt$par[1]
  final_data[i, 3] <- exp(bestopt$par[1])
  final_data[i, 4] <- bestopt$convergence
}

# create data frame
colnames(final_data) <- c("id", "gamma_log", "gamma", "convergence")
parameter_data <- as.data.frame(final_data)