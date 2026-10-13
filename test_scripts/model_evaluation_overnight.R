# Model evaluation on train and test sets ----
# Evaluates one or more fitted sdmTMB models on any number of datasets
# (train and test sets) and returns one row of metrics per model x dataset.

# Libraries ----
library(sdmTMB)
library(arrow)    # read_parquet()
library(tweedie)  # dtweedie(), tweedie.dev()

setwd("/Users/Wanja/Documents/non-equilibrium_data")

# 1. Settings ----
response   <- "Npp"
chunk      <- 5e5   # rows per predict() call (limits memory use)
ll_n       <- 2e5   # rows used for the Tweedie log-likelihood (dtweedie is slow); Inf = all

# 2. Load data ----
# Add or remove names here; everything below adapts.
dir_tab   <- "tables_wgs84_allyears"
set_names <- c("train", "test1", "test2", "test3")

read_set <- function(name) {
  f <- sprintf("table_wgs84_2002-2023_%s_5000px_seed42.parquet", name)
  as.data.frame(read_parquet(file.path(dir_tab, f)))
}
datasets <- setNames(lapply(set_names, read_set), set_names)

# Only needed if the parquet files were NOT already scaled in the other script
# scaling_params <- read.csv(file.path(dir_tab, "scaling_params_train_100000px_seed42.csv"))

# 3. Load models ----
# Named list. Add as many models as you like.
models <- list(
  mod0 = readRDS("output/output_overnight_test/mod0_20261013.rds"),
  mod2 = readRDS("output/output_overnight_test/mod2_20261013.rds")
)

# `datasets` is shared by all models. If models were trained on different train
# sets, make it a list of lists instead, keyed by model name:
# datasets <- list(mod2 = list(train = ..., test1 = ...), mod3 = list(train = ..., test1 = ...))

# 4. Helper functions ----
predict_chunked <- function(fit, newdata, chunk) {
  idx <- split(seq_len(nrow(newdata)), ceiling(seq_len(nrow(newdata)) / chunk))
  unlist(lapply(idx, function(i) {
    predict(fit, newdata = newdata[i, , drop = FALSE], type = "response")$est
  }), use.names = FALSE)
}

evaluate <- function(fit, d, response = "Npp") {
  mu <- predict_chunked(fit, d, chunk)
  y  <- d[[response]]
  ok <- is.finite(mu) & is.finite(y)
  y <- y[ok]; mu <- mu[ok]
  
  rp  <- tidy(fit, "ran_pars")
  phi <- rp$estimate[rp$term == "phi"]
  p   <- rp$estimate[rp$term == "tweedie_p"]
  i   <- if (length(y) > ll_n) sample.int(length(y), ll_n) else seq_along(y)
  
  data.frame(
    loglik = mean(log(dtweedie(y[i], mu = mu[i], phi = phi, power = p)), na.rm = TRUE),
    rmse   = sqrt(mean((y - mu)^2)),
    mae    = mean(abs(y - mu))
  )
}

# 5. Run evaluation ----
# models: named list of fitted models; datasets: named list (train, test1, ...)
set.seed(1)
results <- do.call(rbind, lapply(names(models), function(m) {
  do.call(rbind, lapply(names(datasets), function(s) {
    message(format(Sys.time(), "%H:%M:%S"), "  ", m, " on ", s)
    cbind(model = m, dataset = s, evaluate(models[[m]], datasets[[s]]))
  }))
}))

# 6. Output ----
print(results, digits = 4)

stamp <- format(Sys.Date(), "%Y%m%d")
write.csv(results, sprintf("output/output_overnight_test/model_evaluation_%s.csv", stamp), row.names = FALSE)
