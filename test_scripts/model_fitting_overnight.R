# Model fitting script - overnight


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
    description = "Linear model, space + time"
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
    notif_message, title, sound
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

# -> see model_validation_overnight.R
