# Manual model validation 

# Load libraries
library(sdmTMB)    # tidy, sanity, predict, simulate, residuals, dharma_residuals
library(ggeffects) # predict_response
library(DHARMa)    # residual plots and spatial test
library(ggplot2)   # plots
library(gstat)     # variograms
library(sp)        # coordinates()
library(hexbin)    # geom_hex

# Set the working directory and import the data.
setwd("/Users/Wanja/Documents/non-equilibrium_data")

# Load scaling parameters
scaling_params <- read.csv("tables_wgs84_allyears/scaling_params_train_50000px_seed42.csv")

# Folder with saved models
output_folder <- "output/output_overnight"

# Base R png device: same physical size as before (pixel values / 72 dpi), saved at 300 dpi
png_plot <- function(file, width = 600, height = 400) {
  png(file, width = width / 72, height = height / 72, units = "in", res = 300)
}

# ggplot size matching the base R plots, with matching text sizes (base R default is 12 pt)
gg_width  <- 600 / 72
gg_height <- 400 / 72

theme_plain <- function(base_size = 12) {
  theme_bw(base_size = base_size) +
    theme(panel.grid = element_blank(),
          axis.text = element_text(size = base_size),
          axis.title = element_text(size = base_size),
          legend.text = element_text(size = base_size),
          legend.title = element_text(size = base_size))
}

# Returns a function converting a scaled covariate back to its original units,
# using the mean and sd saved from the training set: x = mean + sd * x_sc
make_unscaler <- function(sc_name, params = scaling_params) {
  orig_name <- sub("_sc$", "", sc_name)
  
  # year_sc: shifted by the minimum year, not z-scored
  if (sc_name == "year_sc") {
    year_min <- params$mean[params$variable == "year_min"]
    return(function(x) x + year_min)
  }
  
  row <- params[params$variable == orig_name, ]
  if (nrow(row) == 1 && !is.na(row$sd)) {
    return(function(x) row$mean + row$sd * x)
  }
  NULL
}


mod_name <- "mod4_wiggly_20261016"
mod <- readRDS(file.path(output_folder, paste0(mod_name, ".rds")))

start_time <- Sys.time()

term = "pr_sum_sc"
vals <- seq(min(mod$data[[term]], na.rm = TRUE),
            max(mod$data[[term]], na.rm = TRUE), length.out = 100)
pred <- predict_response(mod,
                         terms = sprintf("%s [%s]", term, paste(signif(vals, 6), collapse = ", ")),
                         ci_level = NA)
#pred <- predict_response(mod, terms = "pr_sum_sc [all]", ci_level = NA)
print(pred)
plot(pred)

end_time <- Sys.time()
run_time <- end_time - start_time
print(run_time)




library(ggplot2)

mod_name <- "mod4_wiggly_20261016"
mod <- readRDS(file.path(output_folder, paste0(mod_name, ".rds")))

# Prediction grid: one block per covariate, the others held at their mean ----
make_curve_data <- function(mod, terms, n = 100) {
  d    <- mod$data
  tmpl <- d[1, , drop = FALSE]        # keeps coordinates and any other columns
  covs <- terms[terms %in% names(d)]
  base <- sapply(covs, function(v) mean(d[[v]], na.rm = TRUE))
  
  do.call(rbind, lapply(covs, function(v) {
    nd <- tmpl[rep(1, n), , drop = FALSE]
    for (b in covs) nd[[b]] <- base[[b]]
    nd[[v]]    <- seq(min(d[[v]], na.rm = TRUE), max(d[[v]], na.rm = TRUE), length.out = n)
    nd$term    <- v
    nd$x_focal <- nd[[v]]
    nd
  }))
}

model_vars <- setdiff(all.vars(mod$formula[[1]]), "Npp")
nd_all <- make_curve_data(mod, model_vars, n = 100)

# 1) Timing test with 5 rows ----
t5 <- system.time(
  pr5 <- predict(mod, newdata = nd_all[1:5, ], re_form = NA, type = "response")
)
print(t5)
if (t5[["elapsed"]] > 600) stop("5 rows took more than 10 minutes, do not run the full call")

# 2) Full prediction, all covariates in one call ----
start_time <- Sys.time()
pr <- predict(mod, newdata = nd_all, re_form = NA, type = "response")
print(Sys.time() - start_time)

# 3) Split by covariate, unscale the x axis, plot ----
pred_list <- list()
for (term in unique(pr$term)) {
  df <- pr[pr$term == term, ]
  unscale <- make_unscaler(term)
  
  df_pred <- data.frame(
    x_sc      = df$x_focal,
    x         = if (!is.null(unscale)) unscale(df$x_focal) else df$x_focal,
    predicted = df$est
  )
  pred_list[[term]] <- df_pred
  
  p <- ggplot(df_pred, aes(x = x, y = predicted)) +
    geom_line(linewidth = 1) +
    labs(x = if (!is.null(unscale)) sub("_sc$", "", term) else term,
         y = "Predicted Npp") +
    theme_bw()
  print(p)
}




# ---- Visualize smoother based on coefficients ----
output_folder <- "output/output_overnight_test" 
mod_name <- "mod2_20261013"
mod <- readRDS(file.path(output_folder, paste0(mod_name, ".rds")))
str(mod)
visreg(mod, "pr_sum_sc")



d  <- mod$data
td <- mod$tmb_data
pl <- mod$tmb_obj$env$parList(par = mod$tmb_obj$env$last.par.best)

X  <- if (is.list(td$X_ij)) td$X_ij[[1]] else td$X_ij   # fixed-effect design matrix
b  <- as.numeric(pl$b_j)
bs <- as.matrix(pl$bs)

colnames(X)                 # check the column names
names(td$Zs); length(td$Zs) # one entry per smooth
td$b_smooth_start           # where each smooth's coefficients start (0-based)

# Contribution of each smooth: Z_k %*% coefficients_k
smooth_contrib <- lapply(seq_along(td$Zs), function(k) {
  idx <- td$b_smooth_start[k] + seq_len(ncol(td$Zs[[k]]))
  as.numeric(td$Zs[[k]] %*% bs[idx, 1])
})

# 1) CHECK: everything together must match sdmTMB's own fixed-effect predictor
eta_fixed <- as.numeric(X %*% b) + Reduce(`+`, smooth_contrib) +
  (if (!is.null(td$offset_i)) td$offset_i else 0)
p0 <- predict(mod)          # no newdata
print(all.equal(eta_fixed, p0$est_non_rf, tolerance = 1e-6))



df1 <- read.csv("/Users/Wanja/Documents/non-equilibrium_data/output/output_overnight/run_log_20261015.csv")
df2 <- read.csv("/Users/Wanja/Documents/non-equilibrium_data/output/output_overnight/run_log_20261016.csv")
df3 <- read.csv("/Users/Wanja/Documents/non-equilibrium_data/output/output_overnight/run_log_20261017.csv")
df_all <- dplyr::bind_rows(df1, df2, df3)
df_all
df_all_sorted <- df_all[order(df_all$AIC), ]
write.csv(df_all_sorted, "/Users/Wanja/Documents/non-equilibrium_data/output/output_overnight/run_log_all_sorted.csv")
