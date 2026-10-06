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



# Set the working directory and import the data.
setwd("/Users/Wanja/Documents/non-equilibrium_data")

# Load example dataset
df <- read_parquet("cv_new/table_wgs84_2002-2018_with_cv_sample.parquet")

# Check it loaded correctly
dim(df)             # number of rows/columns
str(df)             # columns
names(df)           # column names


# Section 3: Data coding ----

# Create a location grouping factor from lon/lat (each unique coordinate pair = one location)
df$location_id <- interaction(df$longitude, df$latitude, drop = TRUE)
locations <- unique(df$location_id)
length(locations)

# Omit outliers: keep only rows below the 99th percentile for Npp & pr_sum
npp_99 <- quantile(df$Npp, 0.99, na.rm = TRUE)
pr_sum_99 <- quantile(df$pr_sum, 0.99, na.rm = TRUE)
df <- df[df$Npp < npp_99 & df$pr_sum < pr_sum_99, ]
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
output_folder <- "output_overnight"

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
    formula = Npp ~ 1,
    spatial = "on",
    family = tweedie(link = "log"),
    description = "Null model with space"
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
    #mesh = mesh_5000_5     # must already exist in the environment
  )
)


# Run logic

models_to_run <- names(model_registry)[sapply(model_registry, function(x) isTRUE(x$run))]
cat("Models queued for this run:", paste(models_to_run, collapse = ", "), "\n")

run_results <- data.frame()
fitted_models <- list()
coef_results <- list()
sanity_results <- list()

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
    
    coefs <- tryCatch(tidy(mod, effects = "fixed", conf.int = TRUE), error = function(e) NULL)
    coefs_ran <- tryCatch(tidy(mod, effects = "ran_pars", conf.int = TRUE), error = function(e) NULL)
    if (!is.null(coefs)) coefs$name <- mod_name
    if (!is.null(coefs_ran)) coefs_ran$name <- mod_name
    coef_results[[mod_name]] <- list(fixed = coefs, ran_pars = coefs_ran)
    
    converged <- !is.null(mod$sd_report) && mod$sd_report$pdHess
    max_gradient <- tryCatch(max(abs(mod$gradients)), error = function(e) NA)
    aic_val <- tryCatch(AIC(mod), error = function(e) NA)
    
    # ---- sanity() check: just save the returned object directly ----
    sanity_check <- tryCatch({
      sanity(mod)
    }, error = function(e) {
      message("sanity() failed for ", mod_name, ": ", conditionMessage(e))
      NULL
    })
    
    if (!is.null(sanity_check)) {
      sanity_results[[mod_name]] <- sanity_check
      sanity_all_ok <- tryCatch(all(unlist(sanity_check)), error = function(e) NA)
    } else {
      sanity_all_ok <- NA
    }
    
  } else {
    converged <- NA; max_gradient <- NA; aic_val <- NA; sanity_all_ok <- NA
  }
  
  run_results <- rbind(run_results, data.frame(
    name = mod_name,
    description = spec$description,
    data_used = if (!is.null(spec$data)) deparse(substitute(spec$data)) else "df_sample (default)",
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

fixed_coefs_all <- do.call(rbind, lapply(coef_results, function(x) x$fixed))
ran_coefs_all <- do.call(rbind, lapply(coef_results, function(x) x$ran_pars))
write.csv(fixed_coefs_all, file.path(output_folder, paste0("fixed_coefs_", format(Sys.Date(), "%Y%m%d"), ".csv")), row.names = FALSE)
write.csv(ran_coefs_all, file.path(output_folder, paste0("ran_coefs_", format(Sys.Date(), "%Y%m%d"), ".csv")), row.names = FALSE)

# Save the full sanity() results as a single RDS (list of named logical vectors per model)
saveRDS(sanity_results, file.path(output_folder, paste0("sanity_results_", format(Sys.Date(), "%Y%m%d"), ".rds")))

run_results


# Notify once run is finished

notify_macos <- function(title, message, sound = "default") {
  cmd <- sprintf(
    'display notification "%s" with title "%s" sound name "%s"',
    message, title, sound
  )
  system(sprintf("osascript -e '%s'", cmd))
}

# Call this at the end of your overnight script
notify_macos(
  title = "Overnight model run finished",
  message = paste(length(models_to_run), "models completed. Check", summary_path)
)

# read data later: mod$data
# read mesh later: mod$spde
# read sanity later:
# sanity_result <- readRDS(file.path(output_folder, "sanity_results_20261001.rds"))
# sanity_result          # prints nicely if sanity() has its own print/cli method, same as if you ran it fresh
# str(sanity_result)     # see the raw list structure
# all(unlist(sanity_result))  # quick TRUE/FALSE overall check


# Section 7: Model interpretation ----

# Look at an example model
mod2 <- readRDS("output_overnight/mod2_20261001.rds")

# Parameters
tidy(mod2, effects = "ran_pars", conf.int = TRUE)
#'The Tweedie dispersion (phi) and power parameters control the distribution’s 
#'mean-variance relationship. The Matérn range is the distance at which spatial 
#'correlation becomes negligible (~0.13 correlation). The marginal spatial field 
#'standard deviation (sigma_O) represents unexplained spatial variation.

# Basic sanity check
sanity(mod2)

# Plot fitted values
pred <- predict_response(mod2, terms = "pr_sum_sc [all]", ci_level = NA) #[-2:4]
plot(pred)

pred1 <- predict_response(mod2, terms = "tmmx_mean_sc [all]", ci_level = NA) # include [all] in the terms string to get a smooth plot
plot(pred1)

pred2 <- predict_response(mod2, terms = "pr_sum_cv_sc [all]", ci_level = NA)
plot(pred2)

pred3 <- predict_response(mod2, terms = "elevation_mean_sc [all]", ci_level = NA)
plot(pred3)

#' intercept has very large confidence intervals


# Section 8: Model validation ----

#* Subsection 8.0: Extract residuals ----

#' I use two types of residuals:
#'
#'   1. Analytical randomized-quantile residuals - normal distribution
#'   2. Simulation-based randomized-quantile residuals from DHARMa - uniform distribution

# Extract analytical randomized-quantile residuals
r1 <- residuals(mod2, type = "mle-mvn")

# Extract simulation-based randomized-quantile residuals from DHARMa
sim <- simulate(mod2, nsim = 500, type = "mle-mvn")
r2 <- dharma_residuals(sim, mod2, return_DHARMa = TRUE)

#'The randomized quantile residuals in residuals.sdmTMB() are returned such that 
#'they will be normal(0, 1) if the model is consistent with the data. DHARMa residuals, 
#'however, are returned as uniform(0, 1) under those same circumstances. 


#* Subsection 8.1: Residual distribution ----

# DHARMa
hist(r2)
plotQQunif(r2)
# general tests
testResiduals(r2)


#* Subsection 8.2: Residuals vs. fitted values ---- 

# DHARMa
plotResiduals(r2)

#' too large fitted values for large Npp i think

#* Subsection 8.3: Residuals vs. covariates (in and not in the model) ----

# DHARMa: set rank = FALSE??
plotResiduals(r2, form = mod2$data$tmmx_mean_sc)
plotResiduals(r2, form = mod2$data$pr_sum_sc)
plotResiduals(r2, form = mod2$data$pr_sum_cv_sc)
plotResiduals(r2, form = mod2$data$elevation_mean_sc)
plotResiduals(r2, form = mod2$data$vegetation_length_sc)


#* Subsection 8.4: Overdispersion and zero-inflation ----

# DHARMa
testDispersion(r2)
testZeroInflation(r2)


#* Subsection 8.5: Residuals vs. time ----

# DHARMa
plotResiduals(r2, form = mod2$data$year_sc)


# Autocorrelation
set.seed(123)
n_sample_acf <- 20  # how many locations to check
locations_acf_sample <- sample(unique(df_sample$location_id), size = n_sample_acf)

# Attach residuals to df_sample for easy subsetting (safe since r2 was computed directly on df_sample)
df_sample$resid <- r2$scaledResiduals # replace with r1 for analytical residuals 

par(mfrow = c(4, 5), mar = c(4, 4, 2, 1))
for (loc in locations_acf_sample) {
  idx <- df_sample$location_id == loc
  # order by year to ensure correct temporal sequence
  resid_yr <- df_sample$resid[idx][order(df_sample$year[idx])]
  acf(resid_yr, main = paste("Loc:", loc), lag.max = 5)
}
par(mfrow = c(1, 1))


set.seed(123)
n_sample_acf <- 100  # how many locations to check
locations_acf_sample <- sample(unique(df_sample$location_id), size = n_sample_acf)

acf_resid_by_loc <- sapply(locations_acf_sample, function(loc) {
  idx <- df_sample$location_id == loc
  resid_yr <- df_sample$resid[idx][order(df_sample$year[idx])]
  if (length(resid_yr) < 6) return(rep(NA, 5))
  acf(resid_yr, plot = FALSE, lag.max = 5)$acf[2:6]
})

acf_resid_by_loc <- t(acf_resid_by_loc)  # locations x lags
colnames(acf_resid_by_loc) <- paste0("lag", 1:5)

summary(acf_resid_by_loc)

par(mfrow = c(2, 3))
for (i in 1:5) {
  hist(acf_resid_by_loc[,i])
  abline(v = 0, col = "red", lty = 2)
}
par(mfrow = c(1, 1))


# Mean ACF decay curve
mean_acf <- colMeans(acf_resid_by_loc, na.rm = TRUE)
plot(1:5, mean_acf, type = "b", pch = 16,
     xlab = "Lag", ylab = "Mean residual autocorrelation",
     main = "Average temporal ACF decay of residuals")
abline(h = 0, col = "red", lty = 2)

boxplot(acf_resid_by_loc[,1], acf_resid_by_loc[,2], acf_resid_by_loc[,3], acf_resid_by_loc[,4], acf_resid_by_loc[,5])


#* Subsection 8.6: Residuals vs spatial coordinates ----

# calculating x, y positions per group (location)
groupLocations <- aggregate(df_sample[, c("longitude", "latitude")], 
                            list(location_id = df_sample$location_id), mean)

# calculating residuals per group, i.e. summing residuals of all years for each location
r2_agg <- recalculateResiduals(r2, group = df_sample$location_id)

# running the spatial test on grouped residuals
testSpatialAutocorrelation(r2_agg, groupLocations$longitude, groupLocations$latitude)


library(gstat)
library(sp)

# Combine aggregated residuals with their coordinates
resid_df <- data.frame(
  longitude = groupLocations$longitude,
  latitude = groupLocations$latitude,
  residual = r2_agg$scaledResiduals
)

# Convert to a spatial object
coordinates(resid_df) <- ~ longitude + latitude

# Compute and plot the empirical variogram
vg <- variogram(residual ~ 1, data = resid_df)
plot(vg, main = "Variogram of aggregated DHARMa residuals")




years <- sort(unique(df_sample$year))

spatial_test_results <- list()
variogram_results <- list()

for (yr in years) {
  idx <- which(df_sample$year == yr)
  
  resid_yr <- r2$scaledResiduals[idx]
  coords_yr <- df_sample[idx, c("longitude", "latitude")]
  
  # testSpatialAutocorrelation accepts a numeric vector directly, not just a DHARMa object
  spatial_test_results[[as.character(yr)]] <- testSpatialAutocorrelation(
    resid_yr, x = coords_yr$longitude, y = coords_yr$latitude, plot = FALSE
  )
  
  df_vg <- data.frame(longitude = coords_yr$longitude,
                      latitude = coords_yr$latitude,
                      residual = resid_yr)
  coordinates(df_vg) <- ~ longitude + latitude
  variogram_results[[as.character(yr)]] <- variogram(residual ~ 1, data = df_vg)
}

# Moran's I p-value per year
sapply(spatial_test_results, function(x) x$p.value)

# Panel of variogram plots, one per year
n_years <- length(years)
ncol_panel <- 4
nrow_panel <- ceiling(n_years / ncol_panel)

par(mfrow = c(nrow_panel, ncol_panel), mar = c(4, 4, 2, 1))
for (yr in years) {
  vg <- variogram_results[[as.character(yr)]]
  plot(vg$dist, vg$gamma, main = paste("Year:", yr), 
       xlab = "Distance", ylab = "Semivariance", pch = 16)
}
par(mfrow = c(1, 1))

#' there seems to still be unaccounted spatial correlation even with a simple mesh


#* Subsection 8.7: Observed vs. fitted values ----

fitted_vals <- fitted(mod2)
observed_vals <- df_sample$Npp

plot(fitted_vals, observed_vals,
     xlab = "Fitted values", ylab = "Observed values",
     main = "Observed vs. Fitted - mod2",
     pch = 16, col = rgb(0, 0, 0, 0.3))
abline(0, 1, col = "red", lwd = 2, lty = 2)  # 1:1 reference line


df_plot <- data.frame(fitted = fitted_vals, observed = observed_vals)

ggplot(df_plot, aes(x = fitted, y = observed)) +
  geom_hex(bins = 50) +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed", linewidth = 1) +
  scale_fill_viridis_c() +
  labs(x = "Fitted values", y = "Observed values", title = "Observed vs. Fitted - mod2",
       fill = "Count") +
  theme_bw()


df_plot <- data.frame(fitted = fitted_vals, observed = observed_vals)

ggplot(df_plot, aes(x = fitted, y = observed)) +
  geom_smooth(method = "gam", formula = y ~ s(x, bs = "cs"), color = "steelblue", linewidth = 1) +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed", linewidth = 1) +
  labs(x = "Fitted values", y = "Observed values", title = "Observed vs. Fitted - mod2 (smoother)") +
  theme_bw()

#' some very large fitted values for lower observed values. 
#' no overfitting visible


#* Subsection 8.8: Conclusions

#' - there's problems with almost every residual test
#' - main problem: non-linear relations with tmmx, pr, pr_cv (maybe)
#' - elevation and year might only have a linear effect depending on if one looks at the scale or original plot
#' - year needs to be included, but maybe linearly is enough
#' - still spatial correlation with a simple mesh, but better than no mesh; finer mesh makes it only slightly better
#' - maybe if the above is adjusted, the non normality, fitted values, overdispersion and zero inflation will go away
#' - and maybe the tests are just all significant because of the high sample size?

