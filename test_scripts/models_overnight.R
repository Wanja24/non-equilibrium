# Modelling script - overnight


# Section 1: Data description ----


#* Subsection 1.1: Data source / Scientific background ----


#* Subsection 1.2: Aim ----

#  Hypotheses:

#' 1. Precipitation variability in the growing season is a stronger predictor of 
#' ANPP compared to variability of annual precipitation (Fernandez-Gimenez & Allen-Diaz, 1999).
#' → compare models with yearly precipitation & Cv vs. model with veg period precipitation & Cv

#' 2. ANPP responds positively to nominal variation in precipitation and negatively 
#' to extreme precipitation levels (Knapp et al., 2017). 
#' And/or ANPP responds positively to increased precipitation variability in 
#' non-equilibrium sites and negatively in equilibrium sites (Gherardi & Sala, 2019).
#' → compare models with different non-linear relation between precipitation / precipitation CV and Npp

#' 3. There is a threshold of 33% coefficient of variation of precipitation 
#' (von Wehrden et al., 2012) and a similar threshold for the coefficient of 
#' variation of ANPP that act as main determinants of rangeland dynamics.
#' → how to determine the threshold CV pr sum, CV veg pr sum (könnte anders sein 
#' als im Jahr, HvW’s Hypothese: 35%) CV NPP ?? - regression tree 

#' 4. Current global ANPP is lower than in the past at the same mean precipitation 
#' due to grazing-induced degradation, and this effect interacts along the gradient 
#' ranging from equilibrium to non-equilibrium dynamics.
#' → mixed effect model with year as covariate


#  Design:

#' Response: 
#'    - NPP: Npp
#'    - (CV NPP: Npp_cv)
#'    - RUE: rue
#'
#' Main predictors: 
#'    - precipitation: pr_sum, veg_pr_sum
#'    - CV precipitation: pr_sum_cv, veg_pr_sum_cv
#'    
#' Additional predictors:
#'    - temperature: tmmn_mean or tmmx_mean, veg_tmmn_mean or veg_tmmx_mean
#'    - elevation: elevation_mean
#'    - (vegetation period length: vegetation_length)
#'    - year: year
#'    - location: latitude and longitude


# Main questions:

#'    - which predictors significantly explain NPP / RUE?
#'    - are the vegetation period predictors better than the yearly ones?
#'    - are there non-linear effects?
#'    - is there spatial-temporal correlation, i.e. do year and location have an effect?


# Section 2: Import the data ----

# Load libraries
library(arrow)     # for loading parquet
library(corrplot)  # for correlation matrix
library(car)       # for vif
library(mgcv)      # for frequentist GAMs
library(ggplot2)   # for ggplot plots
library(parallel)  # for parallel processing
library(geosphere) # for distance calculations
library(fmesher)   # for creating meshes
library(glmmTMB)   # for frequentist GLMMs
library(sdmTMB)    # for frequentist spatial-temporal GLMMs and GAMMs
library(ggeffects) # for visualising effects
library(DHARMa)    # for model validation
library(caret)     # for saving models
library(gstat)     # for spatial correlation checks
library(sp)        # for spatial correlation checks
library(hexbin)    # for hexbin plot


# Set the working directory and import the data.
setwd("/Users/Wanja/Documents/non-equilibrium_data")

# Load example dataset
df <- read_parquet("tables_wgs84_allyears/table_wgs84_2002-2023_sample_100000px_seed42.parquet")

# Check it loaded correctly
dim(df)             # number of rows/columns
str(df)             # columns
names(df)           # column names


# Section 3: Data coding ----

# Omit outliers: keep only rows below the 99th percentile for Npp
npp_99 <- quantile(df$Npp, 0.99, na.rm = TRUE)
df <- df[df$Npp < npp_99, ]
nrow(df)

# Check and drop missing values
sum(is.na(df))      # missing values overall
colSums(is.na(df))  # missing values by column
df <- subset(df, !is.na(pr_sum)) # drop missing values in pr_sum
colSums(is.na(df))  # remaining missing values by column
dim(df)             # remaining rows

# Check and exclude pixels with missing years
table(df$year) 
n_years <- length(unique(df$year))
n_years
range(df$year)

# Keep only locations with data for all years
years_per_location <- tapply(df$year, df$location_id, function(x) length(unique(x)))
complete_locations <- names(years_per_location)[years_per_location == n_years]
df <- df[df$location_id %in% complete_locations, ]

length(complete_locations)  # how many locations have all 18 years
nrow(df)                    # resulting row count
table(df$year) 

# Rescale year
df$year_sc <- df$year - min(df$year) + 2
head(df[, c("year", "year_sc")], 5)
tail(df[, c("year", "year_sc")], 5)
#' I will use the covariate Year in the models. However, it starts at 2002. 
#' If I include it as beta * Year, then the intercept
#' corresponds to Year = 0, which is far outside the observed range.
#' As a result, the intercept estimate is unnecessarily large (in
#' absolute value) and its standard error is inflated.
#' The solution: Rescale year. 

# TODO: Code space

# Check and convert categorical variables to factors
sapply(df, class)   # check data types
# fine so far bc I do not have categorical covariates at this point

# Standardize the continuous covariates to mean 0 and sd 1
vars <- c("tmmn_mean", "tmmx_mean", "pr_sum", "veg_tmmn_mean", "veg_tmmx_mean", 
          "veg_pr_sum", "vegetation_length", "elevation_mean", "pr_sum_cv", "veg_pr_sum_cv")
for (var in vars) {
  df[[paste0(var, "_sc")]] <- as.numeric(scale(df[[var]]))
}
names(df)

# Order observations
df <- df[order(df$location_id, df$year), ]
rownames(df) <- 1:nrow(df)
head(df[, c("location_id", "latitude", "longitude", "year", "year_sc")])
#' For making auto-correlations functions, observations need to be ordered correctly 
#' (from low to high bc the acf assumes that)


# Section 4: Data exploration - see other scripts ----

#' Conclusions
#' 
#' - Missing values: there are some, but we can exclude them bc we have so much data.
#' - Outliers: there are some esp large values in Npp and pr_sum, so I too only up to the 99th percentile.
#'             there is also a lot of high values bc there is still forested areas, need to go back and exclude more areas
#' - Collinearity: no problem according to VIF if we exclude (veg_)tmmn_mean 
#' - Response distribution: needs a Tweedie distribution
#' - Zero inflation: probably no problem
#' - Covariates over time: Temperature and precipitation show a non-linear pattern over time, but the absolute changes are rather small.
#' - Relationships with the response variable: non-linear (hump-shaped) relation with temperature & precipitation & elevation
#'    less Npp with more pr variability, but need a closer look at lower values
#' - Dependency - temporal correlation: 
#'    There is temporal autocorrelation at lag 1 for approximately 30 % of locations, 
#'    and at lag 2 for some locations. Need to wait and see if this is still in the 
#'    residuals or some covariate can already model this.
#' - Dependency - spatial correlation: can only assess after modelling with residual variogram


# Section 5: Modelling approach ----

#* Section 5.1: Brainstorming ----

#' Possible models:
#' - null model: mod0 <- glm(Npp ~ 1, family = tweedie) or glm(Npp ~ year + spatial_term, family = tweedie)
#' - linear model yearly: mod1 <- glm(Npp ~ tmmx_mean + pr_sum + pr_sum_cv + elevation_mean, family = tweedie)
#' - linear model veg: modveg1 <- glm(Npp ~ veg_tmmx_mean + veg_pr_sum + veg_pr_sum_cv + vegetation_length + elevation_mean, family = tweedie)
#' - linear model with temporal effect yearly/veg: see above + year
#' - linear model with spatial correlation yearly/veg: see above + spatial term
#' - linear model with temporal and spatial yearly/veg: see above + year + spatial term
#' - add non-linear smoothers for variables / or quadratic term for downwards hump shape:
#'    - tmmx
#'    - pr
#'    - pr cv
#'    - elevation
#'    - vegetation length
#'    - year
#' 
#' Modelling approach & libraries:
#' Frequentist: glm/glmmTMB for GLM, mgcv for GAM, sdmTMB for spatial GLMM
#' Bayesian: inlabru for all

#* Section 5.2: Model formulation ----

#' Let Npp_it denote the yearly NPP for observation/location i recorded during year t.
#'
#' Npp is a continuous response variable (larger or equal to 0).
#' We therefore use a Tweedie distribution with a log link:
#'
#'   Npp_it ~ Tweedie(mu_it, phi, p)
#'
#'   E(Npp_it)   = mu_it
#'   Var(Npp_it) = phi * mu_it^p
#'
#' The linear predictor is
#'
#'   log(mu_it) =
#'      intercept
#'      + temperature effect
#'      + precipitation effect
#'      + precipitation variability effect
#'      + elevation effect
#'      (+ vegetation length effect)
#'      + year effect
#'      + spatial term u_i
#'
#' The principal ecological questions are:
#'    - which predictors significantly explain NPP / RUE?
#'    - are the vegetation period predictors better than the yearly ones?
#'    - are there non-linear effects?
#'    - is there spatial-temporal correlation, i.e. do year and location have an effect?
#'
#' We therefore compare a sequence of models that differ in:
#'    - yearly vs. vegetation period
#'    - with or without year and/or spatial correlation or spatial-temporal correlation
#'    - each variable can be either linear or non-linear smoother (+ adjust smoothness & mesh)

# Section 6: Fit models ----

#* Subsection 6.1: Frequentist models ----

# Setup: default sample + mesh (used unless a model specifies its own)
output_folder <- "output/output_overnight"

set.seed(123)
n_sample_locations <- 1000
locations_sample <- sample(unique(df$location_id), size = n_sample_locations)
df_sample <- df[df$location_id %in% locations_sample, ]
nrow(df_sample)
length(unique(df_sample$location_id))

mesh_fm0 <- fm_mesh_2d(loc = unique(df_sample[, c("longitude", "latitude")]), cutoff = 5)
mesh_tmb0 <- make_mesh(df_sample, xy_cols = c("longitude", "latitude"), mesh = mesh_fm0)


# Model registry

model_registry <- list(
  mod0 = list(
    run = TRUE,
    formula = Npp ~ year_sc,
    spatial = "on",
    family = tweedie(link = "log"),
    description = "Null model, space + time"
    # data / mesh omitted -> uses defaults
  ),
  mod1 = list(
    run = FALSE,
    formula = Npp ~ tmmx_mean_sc + pr_sum_sc + pr_sum_cv_sc + elevation_mean_sc + year_sc,
    spatial = "on",
    family = tweedie(link = "log"),
    description = "Linear model yearly, space + time"
  ),
  mod2 = list(
    run = TRUE,
    formula = Npp ~ poly(tmmx_mean_sc, 2) + s(pr_sum_sc, bs = "cr", k = 4) +
      s(pr_sum_cv_sc, bs = "cr", k = 4) + elevation_mean_sc + year_sc,
    spatial = "on",
    family = tweedie(link = "log"),
    description = "Smoothers k=4/4, space + time"
  ),
  mod3 = list(
    run = FALSE,
    formula = Npp ~ poly(tmmx_mean_sc, 2) + s(pr_sum_sc, bs = "cr", k = 4) +
      s(pr_sum_cv_sc, bs = "cr", k = 8) + elevation_mean_sc + year_sc,
    spatial = "on",
    family = tweedie(link = "log"),
    description = "Smoothers k=4/8 (wigglier CV), space + time"
  ),
  # Example of a model using a DIFFERENT sample/mesh than the default
  mod4_bigsample = list(
    run = FALSE,
    formula = Npp ~ poly(tmmx_mean_sc, 2) + s(pr_sum_sc, bs = "cr", k = 4) +
      s(pr_sum_cv_sc, bs = "cr", k = 4) + elevation_mean_sc + year_sc,
    spatial = "on",
    family = tweedie(link = "log"),
    description = "Same as mod2 but on 5000-location sample, cutoff 5"#,
    #data = df_5000,        # must already exist in the environment
    #data_name = "df_5000"
    #mesh = mesh_5000_5     # must already exist in the environment
    #mesh_name = "mesh_5000_5"
  )
)


# Run logic

models_to_run <- names(model_registry)[sapply(model_registry, function(x) isTRUE(x$run))]
cat("Models queued for this run:", paste(models_to_run, collapse = ", "), "\n")

run_results <- data.frame()

for (mod_name in models_to_run) {
  spec <- model_registry[[mod_name]]
  cat("\n--- Fitting", mod_name, ":", spec$description, "---\n")
  
  # Use the model's own data/mesh if provided, otherwise fall back to defaults
  mod_data <- if (!is.null(spec$data)) spec$data else df_sample
  mod_mesh <- if (!is.null(spec$mesh)) spec$mesh else mesh_tmb0
  
  start_time <- Sys.time()
  
  mod <- tryCatch({
    sdmTMB(
      formula = spec$formula,
      spatial = spec$spatial,
      mesh = mod_mesh,
      family = spec$family,
      data = mod_data
    )
  }, error = function(e) {
    message("Model failed: ", mod_name, " -- ", conditionMessage(e))
    NULL
  })
  
  end_time <- Sys.time()
  run_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
  
  if (!is.null(mod)) {
    saveRDS(mod, file = file.path(output_folder, paste0(mod_name, "_", format(Sys.Date(), "%Y%m%d"), ".rds")))
    fitted_models[[mod_name]] <- mod
    
    converged <- !is.null(mod$sd_report) && mod$sd_report$pdHess
    max_gradient <- tryCatch(max(abs(mod$gradients)), error = function(e) NA)
    aic_val <- tryCatch(AIC(mod), error = function(e) NA)
    sanity_all_ok <- tryCatch(all(unlist(sanity(mod))), error = function(e) NA)
  } else {
    converged <- NA; max_gradient <- NA; aic_val <- NA; sanity_all_ok <- NA
  }
    
  run_results <- rbind(run_results, data.frame(
    name = mod_name,
    description = spec$description,
    data_used = if (!is.null(spec$data_name)) spec$data_name else "df_sample (default)",
    mesh_used = if (!is.null(spec$mesh_name)) spec$mesh_name else "mesh_tmb0 (default)",
    run_time_secs = run_time,
    converged = converged,
    max_gradient = max_gradient,
    sanity_ok = sanity_all_ok,
    AIC = aic_val,
    timestamp = Sys.time()
  ))
}


# Save run summary (append, not overwrite)
summary_path <- file.path(output_folder, paste0("run_log_", format(Sys.Date(), "%Y%m%d"), ".csv"))
if (file.exists(summary_path)) {
  write.table(run_results, summary_path, sep = ",", row.names = FALSE,
              col.names = FALSE, append = TRUE)
} else {
  write.csv(run_results, summary_path, row.names = FALSE)
}

run_results


# Notify once run is finished
notify_macos <- function(title, notif_message, sound = "default") {
  cmd <- sprintf(
    'display notification "%s" with title "%s" sound name "%s"',
    message, title, sound
  )
  system(sprintf("osascript -e '%s'", cmd))
}

notify_macos(
  title = "Overnight model run finished",
  notif_message = paste(length(models_to_run), "models completed. Check", summary_path)
)

# read data later: mod$data
# read mesh later: mod$spde
# read sanity later:
# sanity_result <- readRDS(file.path(output_folder, "sanity_results_20261001.rds"))
# sanity_result          # prints nicely if sanity() has its own print/cli method, same as if you ran it fresh
# str(sanity_result)     # see the raw list structure
# all(unlist(sanity_result))  # quick TRUE/FALSE overall check


# Section 7 & 8: Model interpretation & validation ----

run_model_diagnostics <- function(mod, mod_name, output_folder) {
  
  diag_folder <- file.path(output_folder, paste0(mod_name, "_diagnostics"))
  dir.create(diag_folder, showWarnings = FALSE, recursive = TRUE)
  
  # ---- Parameters ----
  ran_pars <- tryCatch(tidy(mod, effects = "ran_pars", conf.int = TRUE), error = function(e) NULL)
  fixed_pars <- tryCatch(tidy(mod, effects = "fixed", conf.int = TRUE), error = function(e) NULL)
  if (!is.null(ran_pars)) write.csv(ran_pars, file.path(diag_folder, "ran_pars.csv"), row.names = FALSE)
  if (!is.null(fixed_pars)) write.csv(fixed_pars, file.path(diag_folder, "fixed_pars.csv"), row.names = FALSE)
  
  # ---- Sanity ----
  sanity_check <- tryCatch(sanity(mod), error = function(e) NULL)
  if (!is.null(sanity_check)) saveRDS(sanity_check, file.path(diag_folder, "sanity.rds"))
  
  # ---- Predicted response curves ----
  pred_terms <- c("tmmx_mean_sc", "pr_sum_sc", "pr_sum_cv_sc", "elevation_mean_sc")
  pred_list <- list()
  
  for (term in pred_terms) {
    pred <- tryCatch(predict_response(mod, terms = paste0(term, " [all]"), ci_level = NA),
                     error = function(e) NULL)
    if (!is.null(pred)) {
      pred_list[[term]] <- pred
      p <- plot(pred) + labs(title = paste(mod_name, "-", term))
      ggsave(file.path(diag_folder, paste0("pred_", term, ".png")), p, width = 6, height = 4)
    }
  }
  saveRDS(pred_list, file.path(diag_folder, "predictions.rds"))
  
  # ---- Residuals ----
  r1 <- tryCatch(residuals(mod, type = "mle-mvn"), error = function(e) NULL)
  if (!is.null(r1)) saveRDS(r1, file.path(diag_folder, "r1_analytical_residuals.rds"))
  
  sim <- tryCatch(simulate(mod, nsim = 500, type = "mle-mvn"), error = function(e) NULL)
  r2 <- if (!is.null(sim)) tryCatch(dharma_residuals(sim, mod, return_DHARMa = TRUE),
                                    error = function(e) NULL) else NULL
  
  if (!is.null(r2)) {
    saveRDS(r2, file.path(diag_folder, "r2_dharma_residuals.rds"))
    
    # Distribution checks
    png(file.path(diag_folder, "r2_histogram.png"), width = 600, height = 400)
    hist(r2)
    dev.off()
    
    png(file.path(diag_folder, "r2_qq_uniform.png"), width = 600, height = 400)
    plotQQunif(r2)
    dev.off()
    
    # Residuals vs fitted
    png(file.path(diag_folder, "r2_resid_vs_fitted.png"), width = 600, height = 400)
    plotResiduals(r2)
    dev.off()
    
    # Residuals vs each covariate
    covariates <- c("tmmx_mean_sc", "pr_sum_sc", "pr_sum_cv_sc", "elevation_mean_sc",
                    "vegetation_length_sc", "year_sc")
    for (cov in covariates) {
      if (cov %in% names(mod$data)) {
        png(file.path(diag_folder, paste0("r2_resid_vs_", cov, ".png")), width = 600, height = 400)
        tryCatch(plotResiduals(r2, form = mod$data[[cov]]), error = function(e) NULL)
        dev.off()
      }
    }
    
    # Dispersion and zero-inflation plots (plots only, test results not saved)
    png(file.path(diag_folder, "r2_dispersion_test.png"), width = 600, height = 400)
    tryCatch(testDispersion(r2), error = function(e) NULL)
    dev.off()
    
    png(file.path(diag_folder, "r2_zeroinflation_test.png"), width = 600, height = 400)
    tryCatch(testZeroInflation(r2), error = function(e) NULL)
    dev.off()
    
    # ---- Temporal autocorrelation of residuals within locations ----
    dat <- mod$data
    dat$resid <- r2$scaledResiduals  # use r1 for analytical residuals
    
    set.seed(123)
    n_sample_acf <- min(100, length(unique(dat$location_id)))
    locations_acf_sample <- sample(unique(dat$location_id), size = n_sample_acf)
    
    acf_resid_by_loc <- sapply(locations_acf_sample, function(loc) {
      idx <- dat$location_id == loc
      resid_yr <- dat$resid[idx][order(dat$year_sc[idx])]
      if (length(resid_yr) < 6) return(rep(NA, 5))
      acf(resid_yr, plot = FALSE, lag.max = 5)$acf[2:6]
    })
    acf_resid_by_loc <- t(acf_resid_by_loc)
    colnames(acf_resid_by_loc) <- paste0("lag", 1:5)
    
    write.csv(acf_resid_by_loc, file.path(diag_folder, "acf_resid_by_location.csv"))
    
    # Histograms per lag
    png(file.path(diag_folder, "acf_hist_by_lag.png"), width = 900, height = 600)
    par(mfrow = c(2, 3))
    for (i in 1:5) {
      hist(acf_resid_by_loc[, i], main = paste("Lag", i), xlab = "ACF")
      abline(v = 0, col = "red", lty = 2)
    }
    par(mfrow = c(1, 1))
    dev.off()
    
    # Mean ACF decay curve
    png(file.path(diag_folder, "acf_mean_decay.png"), width = 600, height = 400)
    plot(1:5, colMeans(acf_resid_by_loc, na.rm = TRUE), type = "b", pch = 16,
         xlab = "Lag", ylab = "Mean residual autocorrelation",
         main = "Average temporal ACF decay of residuals")
    abline(h = 0, col = "red", lty = 2)
    dev.off()
    
    # Boxplot per lag
    png(file.path(diag_folder, "acf_boxplot_by_lag.png"), width = 600, height = 400)
    boxplot(acf_resid_by_loc, xlab = "Lag", ylab = "ACF", main = "Residual ACF by lag")
    abline(h = 0, col = "red", lty = 2)
    dev.off()
    
    
    # ---- Spatial autocorrelation of residuals, per year ----
    years <- sort(unique(dat$year_sc))
    
    moran_results <- data.frame(year = years, observed = NA, expected = NA, p_value = NA)
    variogram_results <- list()
    
    for (i in seq_along(years)) {
      yr <- years[i]
      idx <- which(dat$year_sc == yr)
      
      resid_yr <- r2$scaledResiduals[idx]
      coords_yr <- dat[idx, c("longitude", "latitude")]
      
      # Moran's I
      moran <- tryCatch(
        testSpatialAutocorrelation(resid_yr, x = coords_yr$longitude,
                                   y = coords_yr$latitude, plot = FALSE),
        error = function(e) NULL
      )
      if (!is.null(moran)) {
        moran_results$observed[i] <- moran$statistic["observed"]
        moran_results$expected[i] <- moran$statistic["expected"]
        moran_results$p_value[i]  <- moran$p.value
      }
      
      # Empirical variogram
      df_vg <- data.frame(longitude = coords_yr$longitude,
                          latitude = coords_yr$latitude,
                          residual = resid_yr)
      sp::coordinates(df_vg) <- ~ longitude + latitude
      variogram_results[[as.character(yr)]] <- tryCatch(
        gstat::variogram(residual ~ 1, data = df_vg),
        error = function(e) NULL
      )
    }
    
    write.csv(moran_results, file.path(diag_folder, "spatial_morans_I_by_year.csv"), row.names = FALSE)
    saveRDS(variogram_results, file.path(diag_folder, "spatial_variograms_by_year.rds"))
    
    # Panel of variograms, one per year
    n_years <- length(years)
    ncol_panel <- 4
    nrow_panel <- ceiling(n_years / ncol_panel)
    
    png(file.path(diag_folder, "spatial_variograms_by_year.png"),
        width = 300 * ncol_panel, height = 250 * nrow_panel)
    par(mfrow = c(nrow_panel, ncol_panel), mar = c(4, 4, 2, 1))
    for (yr in years) {
      vg <- variogram_results[[as.character(yr)]]
      if (!is.null(vg)) {
        plot(vg$dist, vg$gamma, main = paste("Year:", yr),
             xlab = "Distance", ylab = "Semivariance", pch = 16)
      } else {
        plot.new(); title(main = paste("Year:", yr, "(failed)"))
      }
    }
    par(mfrow = c(1, 1))
    dev.off()
    
    
    # ---- Observed vs. fitted ----
    fitted_vals <- tryCatch(predict(mod, type = "response")$est, error = function(e) NULL)
    
    if (!is.null(fitted_vals)) {
      df_plot <- data.frame(fitted = fitted_vals, observed = mod$data$Npp)
      
      # Scatter (semi-transparent points)
      png(file.path(diag_folder, "obs_vs_fitted_scatter.png"), width = 600, height = 400)
      plot(df_plot$fitted, df_plot$observed,
           xlab = "Fitted values", ylab = "Observed values",
           main = paste("Observed vs. Fitted -", mod_name),
           pch = 16, col = rgb(0, 0, 0, 0.3))
      abline(0, 1, col = "red", lwd = 2, lty = 2)
      dev.off()
      
      # Hexbin
      p_hex <- ggplot(df_plot, aes(x = fitted, y = observed)) +
        geom_hex(bins = 50) +
        geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed", linewidth = 1) +
        scale_fill_viridis_c() +
        labs(x = "Fitted values", y = "Observed values",
             title = paste("Observed vs. Fitted -", mod_name), fill = "Count") +
        theme_bw()
      ggsave(file.path(diag_folder, "obs_vs_fitted_hex.png"), p_hex, width = 6, height = 4)
      
      # Smoother
      p_smooth <- ggplot(df_plot, aes(x = fitted, y = observed)) +
        geom_smooth(method = "gam", formula = y ~ s(x, bs = "cs"),
                    color = "steelblue", linewidth = 1) +
        geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed", linewidth = 1) +
        labs(x = "Fitted values", y = "Observed values",
             title = paste("Observed vs. Fitted (smoother) -", mod_name)) +
        theme_bw()
      ggsave(file.path(diag_folder, "obs_vs_fitted_smooth.png"), p_smooth, width = 6, height = 4)
    }
  }
  
  invisible(list(ran_pars = ran_pars, fixed_pars = fixed_pars, sanity = sanity_check,
                 predictions = pred_list, r1 = r1, r2 = r2))
}


# Run across all saved models

output_folder <- "output/output_overnight"

model_files <- list.files(output_folder, pattern = "^mod[0-9a-zA-Z_]+\\.rds$", full.names = FALSE)
model_names <- gsub("\\.rds$", "", model_files)

n_ok <- 0
n_failed <- 0

for (mod_name in model_names) {
  cat("\n--- Running diagnostics for", mod_name, "---\n")
  
  mod <- tryCatch(readRDS(file.path(output_folder, paste0(mod_name, ".rds"))), error = function(e) NULL)
  
  if (is.null(mod)) {
    n_failed <- n_failed + 1
    next
  }
  
  result <- tryCatch(
    run_model_diagnostics(mod, mod_name, output_folder),
    error = function(e) {
      message("Diagnostics failed for ", mod_name, ": ", conditionMessage(e))
      NULL
    }
  )
  
  if (is.null(result)) n_failed <- n_failed + 1 else n_ok <- n_ok + 1
}

# Notification
notify_macos(
  title = "Model diagnostics finished",
  notif_message = sprintf("%d models done, %d failed. Output in %s", n_ok, n_failed, output_folder)
)
