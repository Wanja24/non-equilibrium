# Modelling script


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


# Section 4: Data exploration ----

#* Subsection 4.1: Missing values ----

colSums(is.na(df)) # see data coding; fine for now

#* Subsection 4.2: Outliers ----

# Histograms
vars_num <- c(vars, "year", "Npp", "Npp_cv")
vars_num

n <- length(vars_num)
ncol_panel <- 3
nrow_panel <- ceiling(n / ncol_panel)

par(mfrow = c(nrow_panel, ncol_panel), mar = c(4, 4, 2, 1))
for (v in vars_num) {
  hist(df[[v]], main = v, xlab = "", col = "grey70", border = "white", breaks = 20)
}
par(mfrow = c(1, 1))

# Summary
summary(df[,vars_num])

# Outliers present mainly in pr_sum and Npp


#* Subsection 4.3: Collinearity ----

# Select all numeric columns, excluding Npp and year
df_num <- df[, c(vars, "year")]

# Spearman correlation matrix
cor_matrix <- cor(df_num, method = "spearman", use = "complete.obs")
print(cor_matrix)

# Spearman correlation heatmap
corrplot(cor_matrix,
         method = "color",       # colored squares (other options: "circle", "number", "shade")
         type = "upper",         # only show upper triangle (avoids redundant mirror)
         order = "hclust",       # cluster similar variables together — helpful for spotting groups
         tl.col = "black",       # text label color
         tl.srt = 45,            # rotate variable labels for readability
         diag = FALSE,           # hide the diagonal (all 1s, not informative)
         addCoef.col = "black",  # overlay the actual correlation values
         number.cex = 0.7)       # size of the coefficient text

# VIF yearly model
vars_vif <- c("tmmn_mean", 
              "tmmx_mean", 
              "pr_sum", 
              "pr_sum_cv",
              "elevation_mean",
              "year",
              "Npp")
df_vif <- df[, vars_vif]
mod_vif <- lm(Npp ~ ., data = df_vif)
vif(mod_vif)
# no multicollinearity if "tmmn_mean" is excluded

# VIF vegetation period model
vars_vif_veg <- c("veg_tmmn_mean", 
                  "veg_tmmx_mean",
                  "veg_pr_sum",
                  "veg_pr_sum_cv",
                  "vegetation_length",
                  "elevation_mean",
                  "year",
                  "Npp")
df_vif_veg <- df[, vars_vif_veg]
mod_vif_veg <- lm(Npp ~ ., data = df_vif_veg)
vif(mod_vif_veg)
# no multicollinearity if "veg_tmmn_mean" is excluded


#* Subsection 4.4: Response distribution ----

hist(df$Npp)
hist(sqrt(df$Npp))
summary(df$Npp)
# continuous right-skewed distribution with zeros: tweedie distribution


#* Subsection 4.5: Zero inflation ----

sum(df$Npp == 0) # Npp contains a few zeros
100 * sum(df$Npp == 0) / nrow(df) # percentage of zeros is < 1%
# the model will most likely be able to deal with these zeros

#* Subsection 4.6: Covariates over time ----

df_veg_clean <- df[!is.na(df$veg_tmmn_mean), ]
dim(df_veg_clean)

ggplot(data = df, aes(y = tmmn_mean, x = year)) + geom_smooth(se = TRUE)
ggplot(data = df, aes(y = tmmx_mean, x = year)) + geom_smooth(se = TRUE)
ggplot(data = df, aes(y = pr_sum, x = year)) + geom_smooth(se = TRUE)
ggplot(data = df, aes(y = pr_sum_cv, x = year)) + geom_smooth(se = TRUE)
ggplot(data = df, aes(y = elevation_mean, x = year)) + geom_smooth(se = TRUE)

ggplot(data = df_veg_clean, aes(y = veg_tmmn_mean, x = year)) + geom_smooth(se = TRUE)
ggplot(data = df_veg_clean, aes(y = veg_tmmx_mean, x = year)) + geom_smooth(se = TRUE)
ggplot(data = df_veg_clean, aes(y = veg_pr_sum, x = year)) + geom_smooth(se = TRUE)
ggplot(data = df_veg_clean, aes(y = veg_pr_sum_cv, x = year)) + geom_smooth(se = TRUE)
ggplot(data = df_veg_clean, aes(y = vegetation_length, x = year)) + geom_smooth(se = TRUE)

# Temperature and precipitation show a non-linear pattern over time, 
# both in the year and in the vegetation period. Temperature increases peaking 
# at 2016, precipitation has a spike at 2010.
# Vegetation length dips in 2010, then increases with temperature.
# But the absolute differences are not large, look at the y-axis scale!


#* Subsection 4.7: Relationships with the response variable ----

ggplot(data = df, aes(y = Npp, x = tmmn_mean)) + geom_smooth(se = TRUE)
ggplot(data = df, aes(y = Npp, x = tmmx_mean)) + geom_smooth(se = TRUE)
ggplot(data = df, aes(y = Npp, x = pr_sum)) + geom_smooth(se = TRUE)
ggplot(data = df, aes(y = Npp, x = pr_sum_cv)) + geom_smooth(se = TRUE)
ggplot(data = df, aes(y = Npp, x = elevation_mean)) + geom_smooth(se = TRUE)

ggplot(data = df_veg_clean, aes(y = Npp, x = veg_tmmn_mean)) + geom_smooth(se = TRUE)
ggplot(data = df_veg_clean, aes(y = Npp, x = veg_tmmx_mean)) + geom_smooth(se = TRUE)
ggplot(data = df_veg_clean, aes(y = Npp, x = veg_pr_sum)) + geom_smooth(se = TRUE)
ggplot(data = df_veg_clean, aes(y = Npp, x = veg_pr_sum_cv)) + geom_smooth(se = TRUE)
ggplot(data = df_veg_clean, aes(y = Npp, x = vegetation_length)) + geom_smooth(se = TRUE)

ggplot(data = df, aes(y = Npp, x = year)) + geom_smooth(se = TRUE)

# plot(df$Npp, df$pr_sum)

# more Npp over time
# non-linear (hump-shaped) relation with temperature & precipitation & elevation
# less Npp with more pr variability, but need a closer look at lower values


#* Subsection 4.8: Dependency - temporal correlation ----

# Plot Npp over time: increasing trend
ggplot(data = df, aes(y = Npp, x = year)) + geom_smooth(se = TRUE)


# Take a random sample of locations
set.seed(123)
n_sample <- 500
locations_sample <- sample(unique(df$location_id), size = n_sample)

ncores <- 6

# Compute autocorrelation for lags 1-5 for each of the sampled locations
acf_list <- mclapply(locations_sample, function(loc) {
  x <- df$Npp[df$location_id == loc]
  x <- x[!is.na(x)]
  if (length(x) < 6) return(rep(NA, 5))  # need enough points for lag-5
  acf(x, plot = FALSE, lag.max = 5)$acf[2:6]  # lags 1 through 5
}, mc.cores = ncores)

# Combine into a matrix: rows = locations, columns = lags 1-5
acf_by_loc <- do.call(rbind, acf_list)
rownames(acf_by_loc) <- locations_sample
colnames(acf_by_loc) <- paste0("lag", 1:5)

# Summary per lag
summary(acf_by_loc)

# Panel of histograms, one per lag
par(mfrow = c(2, 3), mar = c(4, 4, 2, 1))
for (lag in 1:5) {
  hist(acf_by_loc[, lag], breaks = 30, col = "grey70", border = "white",
       main = paste0("Lag ", lag),
       xlab = "ACF")
  abline(v = 0, col = "red", lty = 2)
}
par(mfrow = c(1, 1))


# Average autocorrelation for different lags
mean_acf <- colMeans(acf_by_loc, na.rm = TRUE)
se_acf <- apply(acf_by_loc, 2, sd, na.rm = TRUE) / sqrt(colSums(!is.na(acf_by_loc)))

plot(1:5, mean_acf, type = "b", pch = 16, ylim = c(min(mean_acf - 2*se_acf), max(mean_acf + 2*se_acf)),
     xlab = "Lag", ylab = "Mean autocorrelation across locations",
     main = "Average temporal ACF decay")
arrows(1:5, mean_acf - 1.96*se_acf, 1:5, mean_acf + 1.96*se_acf,
       angle = 90, code = 3, length = 0.05)
abline(h = 0, col = "red", lty = 2)

# Compute autocorrelation plot for 6 random locations
set.seed(12)
n_panels <- 6  # how many locations to show
locations_sample <- sample(unique(df$location_id), size = n_panels)

par(mfrow = c(2, 3), mar = c(4, 4, 2, 1))
for (loc in locations_sample) {
  x <- df$Npp[df$location_id == loc]
  x <- x[!is.na(x)]
  if (length(x) >= 3) {
    acf(x, main = paste("Location:", loc), lag.max = 5)
  }
}
par(mfrow = c(1, 1))  # reset layout

#' There is temporal autocorrelation at lag 1 for approximately 30 % of locations, 
#' and at lag 2 for some locations. Need to wait and see if this is still in the 
#' residuals or some covariate can already model this.


#* Subsection 4.9: Dependency - spatial correlation ----

# Take a random sample of locations
set.seed(123)
n_sample <- 10000
locations_sample <- sample(unique(df$location_id), size = n_sample)

# Get unique lon/lat coordinates for the sampled locations
coords_sample <- unique(df[df$location_id %in% locations_sample, c("location_id", "longitude", "latitude")])
coords_sample <- coords_sample[match(locations_sample, coords_sample$location_id), ]  # keep sample order

# distm() computes a full pairwise distance matrix using the given distance function
dist_matrix <- distm(coords_sample[, c("longitude", "latitude")], fun = distHaversine)

# Add location IDs as row/column names for readability
rownames(dist_matrix) <- coords_sample$location_id
colnames(dist_matrix) <- coords_sample$location_id

# Distances are in meters by default -- convert to km if preferred
dist_matrix_km <- dist_matrix / 1000

dist_matrix_km[1:5, 1:5]  # preview

# Convert to long format using base R
dist_long <- as.data.frame(as.table(dist_matrix_km))
names(dist_long) <- c("location_1", "location_2", "distance_km")

# Remove self-pairs (distance = 0) and duplicate pairs (matrix is symmetric)
dist_long <- dist_long[as.character(dist_long$location_1) < as.character(dist_long$location_2), ]

head(dist_long)

hist(dist_long$distance_km, breaks = 30, col = "grey70", border = "white",
     main = "Distribution of pairwise distances between sampled locations",
     xlab = "Distance (km)")


# Example mesh for 10000 locations
mesh0 <- fm_mesh_2d(loc = coords_sample[, c("longitude", "latitude")] , max.edge=100, cutoff = 1)
mesh0
plot(mesh0)
points(coords_sample[, c("longitude", "latitude")], col = "red", pch = 16, cex = 0.2)

#* Subsection 4.10: Conclusions ----

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

# Take a random sample of locations (all years for each sampled location)
set.seed(123)
n_sample_locations <- 5000  # adjust as needed
locations_sample <- sample(unique(df$location_id), size = n_sample_locations)
df_sample <- df[df$location_id %in% locations_sample, ]
nrow(df_sample)                     # resulting row count
length(unique(df_sample$location_id))  # should equal n_sample_locations

# Make a mesh
# fmesher mesh
mesh_fm0 <- fm_mesh_2d(loc = unique(df_sample[, c("longitude", "latitude")]), cutoff = 1) # before: cutoff 10
mesh_fm0
plot(mesh_fm0)
points(df_sample[, c("longitude", "latitude")], col = "red", pch = 16, cex = 0.2)

# sdmTMB mesh
mesh_tmb0 <- make_mesh(df_sample, xy_cols = c("longitude", "latitude"), mesh = mesh_fm0)
plot(mesh_tmb0)



# Tweedie GLM with glmmTMB/sdmTMB

# Null model
start_time <- Sys.time()

mod0 <- sdmTMB(
  Npp ~ 1,
  spatial = "off",
  mesh = mesh_tmb0,
  family = tweedie(link = 'log'),
  data = df_sample
)
summary(mod0)

end_time <- Sys.time()
run_time <- end_time - start_time
print(run_time)


# Null model with space and time
start_time <- Sys.time()

mod00 <- sdmTMB(
  Npp ~ 1,
  spatial = "on",
  mesh = mesh_tmb0,
  time = "year_sc",
  spatiotemporal = "iid",
  family = tweedie(link = "log"),
  data = df_sample
)
summary(mod00)

end_time <- Sys.time()
run_time <- end_time - start_time
print(run_time)




# Linear model yearly glmmTMB
start_time <- Sys.time()

mod1 <- glmmTMB(
  Npp ~ tmmx_mean_sc + pr_sum_sc + pr_sum_cv_sc + elevation_mean_sc, 
  family = tweedie(link = "log"), REML = TRUE, data = df_sample)

end_time <- Sys.time()
run_time <- end_time - start_time
print(run_time)

summary(mod1)
family_params(mod1)

# Linear model yearly sdmTMB
start_time <- Sys.time()

mod11 <- sdmTMB(
  Npp ~ tmmx_mean_sc + pr_sum_sc + pr_sum_cv_sc + elevation_mean_sc,
  spatial = "off",
  mesh = mesh_tmb0,
  family = tweedie(link = 'log'),
  data = df_sample
)
summary(mod11)

end_time <- Sys.time()
run_time <- end_time - start_time
print(run_time)


# Linear model yearly sdmTMB with space
start_time <- Sys.time()

mod2 <- sdmTMB(
  Npp ~ tmmx_mean_sc + pr_sum_sc + pr_sum_cv_sc + elevation_mean_sc,
  spatial = "on",
  mesh = mesh_tmb0,
  family = tweedie(link = 'log'),
  data = df_sample
)
summary(mod2)

end_time <- Sys.time()
run_time <- end_time - start_time
print(run_time)


# Linear model yearly sdmTMB with space and time
start_time <- Sys.time()

mod3 <- sdmTMB(
  Npp ~ tmmx_mean_sc + pr_sum_sc + pr_sum_cv_sc + elevation_mean_sc + year_sc,
  spatial = "on",
  mesh = mesh_tmb0,
  family = tweedie(link = 'log'),
  data = df_sample
)
summary(mod3)

end_time <- Sys.time()
run_time <- end_time - start_time
print(run_time)


# Linear model yearly sdmTMB with space and time and smoothers
# bs="cr". These have a cubic spline basis defined by a modest sized set of knots spread evenly through the covariate values.
# anything is computationally more efficient than the default thin plate regression splines "tp"
# TODO: choose k (upper limit on the degrees of freedom associated with an s smooth)
# k between 4-8 seems reasonable, 4 is very smooth, 6 moderate and 8 more wiggly (no AIC and residual improvements tho)
start_time <- Sys.time()

mod4 <- sdmTMB(
  Npp ~ poly(tmmx_mean_sc, 2) + s(pr_sum_sc, bs = "cr", k = 4) + s(pr_sum_cv_sc, bs = "cr" , k = 4) + elevation_mean_sc + year_sc,
  spatial = "on",
  mesh = mesh_tmb0,
  family = tweedie(link = 'log'),
  data = df_sample
)
summary(mod4)

end_time <- Sys.time()
run_time <- end_time - start_time
print(run_time)




# Linear model yearly sdmTMB with space: Version with only 1 year
df_sample_2002 <- subset(df_sample, year == 2002)
dim(df_sample)
dim(df_sample_2002)

# Make a mesh
# fmesher mesh
mesh_fm0_2002 <- fm_mesh_2d(loc = unique(df_sample_2002[, c("longitude", "latitude")]), cutoff = 10)
# sdmTMB mesh
mesh_tmb0_2002 <- make_mesh(df_sample_2002, xy_cols = c("longitude", "latitude"), mesh = mesh_fm0_2002)
plot(mesh_tmb0_2002)

mod2_2002 <- sdmTMB(
  Npp ~ tmmx_mean_sc + pr_sum_sc + pr_sum_cv_sc + elevation_mean_sc,
  spatial = "on",
  mesh = mesh_tmb0_2002,
  family = tweedie(link = 'log'),
  data = df_sample_2002
)
summary(mod2_2002)


# Save models
# saveRDS(mod4, "output_models/yearly_model_smoothtemp-smoothpr-smoothcv_space_time.RDS")


#* Subsection 6.2: Track runtime ----

# Take a random sample of locations (all years for each sampled location)
take_random_sample <- function(df, n_locs, seed = 123) {
  set.seed(seed)
  locs_sample <- sample(unique(df$location_id), size = n_locs)
  df_n <- df[df$location_id %in% locs_sample, ]
  print(nrow(df_n))                       # resulting row count
  print(length(unique(df_n$location_id))) # should equal n_locs
  return (df_n)
}

df_100 <- take_random_sample(df, 100)
df_500 <- take_random_sample(df, 500)
df_1000 <- take_random_sample(df, 1000)
df_5000 <- take_random_sample(df, 5000)
df_10000 <- take_random_sample(df, 10000)

# Make a mesh
create_mesh <- function(df, cutoff_value) {
  mesh_fm <- fm_mesh_2d(loc = unique(df[, c("longitude", "latitude")]), cutoff = cutoff_value) # before: cutoff 10
  print(mesh_fm)
  plot(mesh_fm)
  points(df[, c("longitude", "latitude")], col = "red", pch = 16, cex = 0.2)
  mesh_tmb <- make_mesh(df, xy_cols = c("longitude", "latitude"), mesh = mesh_fm)
  return (mesh_tmb)
}

mesh_500_1 <- create_mesh(df_500, 1)
mesh_500_5 <- create_mesh(df_500, 5)
mesh_500_10 <- create_mesh(df_500, 10)
mesh_1000_1 <- create_mesh(df_1000, 1)
mesh_1000_5 <- create_mesh(df_1000, 5)
mesh_1000_10 <- create_mesh(df_1000, 10)
mesh_100_5 <- create_mesh(df_100, 5)
mesh_5000_5 <- create_mesh(df_5000, 5)
mesh_10000_5 <- create_mesh(df_10000, 5)


# Models to compare smoothers

# Define all model configurations in one table
model_specs <- data.frame(
  name = c("no_smoother_nospacetime", "s1_nospacetime", "s2_nospacetime", "s3_nospacetime",
           "no_smoother_spacetime", "s1_spacetime", "s2_spacetime", "s3_spacetime"),
  formula = c(
    "Npp ~ tmmx_mean_sc + pr_sum_sc + pr_sum_cv_sc + elevation_mean_sc",
    "Npp ~ s(tmmx_mean_sc) + pr_sum_sc + pr_sum_cv_sc + elevation_mean_sc",
    "Npp ~ s(tmmx_mean_sc) + s(pr_sum_sc) + pr_sum_cv_sc + elevation_mean_sc",
    "Npp ~ s(tmmx_mean_sc) + s(pr_sum_sc) + s(pr_sum_cv_sc) + elevation_mean_sc",
    "Npp ~ tmmx_mean_sc + pr_sum_sc + pr_sum_cv_sc + elevation_mean_sc + year_sc",
    "Npp ~ s(tmmx_mean_sc) + pr_sum_sc + pr_sum_cv_sc + elevation_mean_sc + year_sc",
    "Npp ~ s(tmmx_mean_sc) + s(pr_sum_sc) + pr_sum_cv_sc + elevation_mean_sc + year_sc",
    "Npp ~ s(tmmx_mean_sc) + s(pr_sum_sc) + s(pr_sum_cv_sc) + elevation_mean_sc + year_sc"
  ),
  spatial = c("off", "off", "off", "off", "on", "on", "on", "on"),
  n_smoothers = c(0, 1, 2, 3, 0, 1, 2, 3),
  stringsAsFactors = FALSE
)
model_specs_quad <- data.frame(
  name = c("quad_tmmx_nospacetime", "quad_tmmx_spacetime"),
  formula = c(
    "Npp ~ poly(tmmx_mean_sc, 2) + s(pr_sum_sc) + s(pr_sum_cv_sc) + elevation_mean_sc",
    "Npp ~ poly(tmmx_mean_sc, 2) + s(pr_sum_sc) + s(pr_sum_cv_sc) + elevation_mean_sc + year_sc"
  ),
  spatial = c("off", "on"),
  n_smoothers = c(2, 2),  # counting only the true s() terms, not the quadratic
  stringsAsFactors = FALSE
)
model_specs <- rbind(model_specs, model_specs_quad)

# Storage
smoother_times <- data.frame()
fitted_models <- list()
coef_list <- list()

for (i in seq_len(nrow(model_specs))) {
  spec <- model_specs[i, ]
  cat("Fitting:", spec$name, "\n")
  
  start_time <- Sys.time()
  
  mod <- tryCatch({
    sdmTMB(
      formula = as.formula(spec$formula),
      spatial = spec$spatial,
      mesh = mesh_500_5,
      family = tweedie(link = "log"),
      data = df_500
    )
  }, error = function(e) {
    message("Model failed: ", spec$name, " -- ", conditionMessage(e))
    NULL
  })
  
  end_time <- Sys.time()
  run_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
  
  # Save the model object to disk
  if (!is.null(mod)) {
    #saveRDS(mod, file = file.path("output_models", paste0(spec$name, "_v2.rds")))
    fitted_models[[spec$name]] <- mod
    
    # Extract coefficients/parameters via sdmTMB's tidy()
    coefs <- tryCatch(tidy(mod, effects = "fixed", conf.int = TRUE), error = function(e) NULL)
    coefs_ran <- tryCatch(tidy(mod, effects = "ran_pars", conf.int = TRUE), error = function(e) NULL)
    
    if (!is.null(coefs)) coefs$name <- spec$name
    if (!is.null(coefs_ran)) coefs_ran$name <- spec$name
    
    coef_list[[spec$name]] <- list(fixed = coefs, ran_pars = coefs_ran)
    
    converged <- !is.null(mod$sd_report) && mod$sd_report$pdHess
    max_gradient <- tryCatch(max(abs(mod$gradients)), error = function(e) NA)
    aic_val <- tryCatch(AIC(mod), error = function(e) NA)
  } else {
    converged <- NA
    max_gradient <- NA
    aic_val <- NA
  }
  
  # Record summary row
  smoother_times <- rbind(smoother_times, data.frame(
    name = spec$name,
    formula = spec$formula,
    spatial = spec$spatial,
    n_smoothers = spec$n_smoothers,
    run_time_secs = run_time,
    converged = converged,
    max_gradient = max_gradient,
    AIC = aic_val
  ))
}

# Save the summary table
#write.csv(smoother_times, "output_models/smoother_comparison_v2.csv", row.names = FALSE)

# Save all coefficients in one combined table
fixed_coefs_all <- do.call(rbind, lapply(coef_list, function(x) x$fixed))
#write.csv(fixed_coefs_all, "output_models/smoother_coefficients_v2.csv", row.names = FALSE)

smoother_times
fixed_coefs_all


# Add a "group" column distinguishing spacetime vs no spacetime, and an x-axis label
smoother_times$group <- ifelse(grepl("spacetime$", smoother_times$name) & !grepl("nospacetime$", smoother_times$name),
                               "Space + Time", "No Space/Time")

# Create a readable x-axis label and explicit ordering, placing quad between 2 and 3 smoothers
smoother_times$x_label <- dplyr::case_when(
  grepl("^no_smoother", smoother_times$name) ~ "0 smoothers",
  grepl("^s1_", smoother_times$name)         ~ "1 smoother",
  grepl("^s2_", smoother_times$name)         ~ "2 smoothers",
  grepl("^quad_", smoother_times$name)       ~ "Quadratic (tmmx)",
  grepl("^s3_", smoother_times$name)         ~ "3 smoothers",
  TRUE ~ smoother_times$name
)

x_order <- c("0 smoothers", "1 smoother", "2 smoothers", "Quadratic (tmmx)", "3 smoothers")
smoother_times$x_label <- factor(smoother_times$x_label, levels = x_order)

ggplot(smoother_times, aes(x = x_label, y = run_time_secs)) +
  geom_col(fill = "steelblue") +
  facet_wrap(~ group, scales = "free_y") +
  labs(x = NULL, y = "Run time (seconds)",
       title = "Model run time by number of smoothers") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


# Models to compare time

# Define model configurations: no time, year_sc as covariate, year_sc as time component
time_specs <- data.frame(
  name = c("no_time", "year_as_covariate", "year_as_time"),
  formula = c(
    "Npp ~ tmmx_mean_sc + pr_sum_sc + pr_sum_cv_sc + elevation_mean_sc",
    "Npp ~ tmmx_mean_sc + pr_sum_sc + pr_sum_cv_sc + elevation_mean_sc + year_sc",
    "Npp ~ tmmx_mean_sc + pr_sum_sc + pr_sum_cv_sc + elevation_mean_sc"
  ),
  use_time_arg = c(FALSE, FALSE, TRUE),
  stringsAsFactors = FALSE
)

time_results <- data.frame()
time_models <- list()
time_coef_list <- list()

for (i in seq_len(nrow(time_specs))) {
  spec <- time_specs[i, ]
  cat("Fitting:", spec$name, "\n")
  
  start_time <- Sys.time()
  
  mod <- tryCatch({
    if (spec$use_time_arg) {
      sdmTMB(
        formula = as.formula(spec$formula),
        time = "year_sc",
        spatiotemporal = "iid",
        spatial = "on",
        mesh = mesh_500_5,
        family = tweedie(link = "log"),
        data = df_500
      )
    } else {
      sdmTMB(
        formula = as.formula(spec$formula),
        spatial = "on",
        mesh = mesh_500_5,
        family = tweedie(link = "log"),
        data = df_500
      )
    }
  }, error = function(e) {
    message("Model failed: ", spec$name, " -- ", conditionMessage(e))
    NULL
  })
  
  end_time <- Sys.time()
  run_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
  
  if (!is.null(mod)) {
    #saveRDS(mod, file = file.path("output_models", paste0(spec$name, ".rds")))
    time_models[[spec$name]] <- mod
    
    coefs <- tryCatch(tidy(mod, effects = "fixed", conf.int = TRUE), error = function(e) NULL)
    coefs_ran <- tryCatch(tidy(mod, effects = "ran_pars", conf.int = TRUE), error = function(e) NULL)
    
    if (!is.null(coefs)) coefs$name <- spec$name
    if (!is.null(coefs_ran)) coefs_ran$name <- spec$name
    
    time_coef_list[[spec$name]] <- list(fixed = coefs, ran_pars = coefs_ran)
    
    converged <- !is.null(mod$sd_report) && mod$sd_report$pdHess
    max_gradient <- tryCatch(max(abs(mod$gradients)), error = function(e) NA)
    aic_val <- tryCatch(AIC(mod), error = function(e) NA)
  } else {
    converged <- NA
    max_gradient <- NA
    aic_val <- NA
  }
  
  time_results <- rbind(time_results, data.frame(
    name = spec$name,
    formula = spec$formula,
    use_time_arg = spec$use_time_arg,
    run_time_secs = run_time,
    converged = converged,
    max_gradient = max_gradient,
    AIC = aic_val
  ))
}

# Save summary table and coefficients
#write.csv(time_results, "output_models/time_comparison.csv", row.names = FALSE)

fixed_coefs_all <- do.call(rbind, lapply(time_coef_list, function(x) x$fixed))
#write.csv(fixed_coefs_all, "output_models/time_coefficients.csv", row.names = FALSE)

time_results

time_results$x_label <- factor(time_results$name,
                               levels = c("no_time", "year_as_covariate", "year_as_time"),
                               labels = c("No time", "Year as covariate", "Year as time component"))

ggplot(time_results, aes(x = x_label, y = run_time_secs)) +
  geom_col(fill = "steelblue") +
  labs(x = NULL, y = "Run time (seconds)",
       title = "Model run time: time handling comparison") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))


# Models to compare spatial meshs

# Define mesh configurations, paired with their correct dataset
mesh_specs <- data.frame(
  name = c("df500_mesh_cutoff1", "df500_mesh_cutoff5", "df500_mesh_cutoff10",
           "df1000_mesh_cutoff1", "df1000_mesh_cutoff5", "df1000_mesh_cutoff10",
           "df500_nomesh", "df1000_nomesh"),
  dataset = c("df_500", "df_500", "df_500", "df_1000", "df_1000", "df_1000",
              "df_500", "df_1000"),
  mesh_name = c("mesh_500_1", "mesh_500_5", "mesh_500_10",
                "mesh_1000_1", "mesh_1000_5", "mesh_1000_10",
                "mesh_500_1", "mesh_1000_5"),   # placeholders, not used when spatial = "off"
  cutoff = c(1, 5, 10, 1, 5, 10, NA, NA),
  spatial = c(rep("on", 6), "off", "off"),
  stringsAsFactors = FALSE
)

mesh_formula <- "Npp ~ tmmx_mean_sc + pr_sum_sc + pr_sum_cv_sc + elevation_mean_sc + year_sc"

mesh_results <- data.frame()
mesh_models <- list()
mesh_coef_list <- list()

for (i in seq_len(nrow(mesh_specs))) {
  spec <- mesh_specs[i, ]
  cat("Fitting:", spec$name, "\n")
  
  # Look up the actual data frame and mesh objects by name
  data_obj <- get(spec$dataset)
  mesh_obj <- get(spec$mesh_name)
  
  # Record number of mesh vertices for reference
  n_mesh_vertices <- if (spec$spatial == "off") 0 else tryCatch(mesh_obj$mesh$n, error = function(e) NA)
  
  start_time <- Sys.time()
  
  mod <- tryCatch({
    sdmTMB(
      formula = as.formula(mesh_formula),
      spatial = spec$spatial,
      mesh = mesh_obj,
      family = tweedie(link = "log"),
      data = data_obj
    )
  }, error = function(e) {
    message("Model failed: ", spec$name, " -- ", conditionMessage(e))
    NULL
  })
  
  end_time <- Sys.time()
  run_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
  
  if (!is.null(mod)) {
    saveRDS(mod, file = file.path("output_models", paste0(spec$name, "_v1.rds")))
    mesh_models[[spec$name]] <- mod
    
    coefs <- tryCatch(tidy(mod, effects = "fixed", conf.int = TRUE), error = function(e) NULL)
    coefs_ran <- tryCatch(tidy(mod, effects = "ran_pars", conf.int = TRUE), error = function(e) NULL)
    
    if (!is.null(coefs)) coefs$name <- spec$name
    if (!is.null(coefs_ran)) coefs_ran$name <- spec$name
    
    mesh_coef_list[[spec$name]] <- list(fixed = coefs, ran_pars = coefs_ran)
    
    converged <- !is.null(mod$sd_report) && mod$sd_report$pdHess
    max_gradient <- tryCatch(max(abs(mod$gradients)), error = function(e) NA)
    aic_val <- tryCatch(AIC(mod), error = function(e) NA)
  } else {
    converged <- NA
    max_gradient <- NA
    aic_val <- NA
  }
  
  mesh_results <- rbind(mesh_results, data.frame(
    name = spec$name,
    dataset = spec$dataset,
    mesh_name = spec$mesh_name,
    cutoff = spec$cutoff,
    spatial = spec$spatial,
    n_mesh_vertices = n_mesh_vertices,
    run_time_secs = run_time,
    converged = converged,
    max_gradient = max_gradient,
    AIC = aic_val
  ))
}

write.csv(mesh_results, "output_models/mesh_comparison_v1.csv", row.names = FALSE)

fixed_coefs_all <- do.call(rbind, lapply(mesh_coef_list, function(x) x$fixed))
write.csv(fixed_coefs_all, "output_models/mesh_coefficients_v1.csv", row.names = FALSE)

mesh_results

mesh_results$cutoff_label <- factor(
  ifelse(is.na(mesh_results$cutoff), "No spatial field", paste("Cutoff =", mesh_results$cutoff)),
  levels = c("No spatial field", "Cutoff = 1", "Cutoff = 5", "Cutoff = 10")
)

ggplot(mesh_results, aes(x = cutoff_label, y = run_time_secs)) +
  geom_col(fill = "steelblue") +
  facet_wrap(~ dataset, scales = "free_y") +
  labs(x = NULL, y = "Run time (seconds)",
       title = "Model run time by mesh cutoff and dataset size") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

ggplot(mesh_results, aes(x = n_mesh_vertices, y = run_time_secs, color = dataset)) +
  geom_point(size = 3) +
  geom_line(aes(group = dataset)) +
  labs(x = "Number of mesh vertices", y = "Run time (seconds)",
       title = "Run time vs. mesh complexity",
       color = "Dataset") +
  theme_bw()


# Compare different number of locations

# Define dataset size configurations, paired with their cutoff-5 mesh
size_specs <- data.frame(
  name = c("df100_cutoff5", "df500_cutoff5", "df1000_cutoff5", "df5000_cutoff5"),
  dataset = c("df_100", "df_500", "df_1000", "df_5000"),
  mesh_name = c("mesh_100_5", "mesh_500_5", "mesh_1000_5", "mesh_5000_5"),
  n_locations = c(100, 500, 1000, 5000),
  stringsAsFactors = FALSE
)

size_formula <- "Npp ~ s(tmmx_mean_sc) + s(pr_sum_sc) + s(pr_sum_cv_sc) + elevation_mean_sc + year_sc"

size_results <- data.frame()
size_models <- list()
size_coef_list <- list()

for (i in seq_len(nrow(size_specs))) {
  spec <- size_specs[i, ]
  cat("Fitting:", spec$name, "\n")
  
  data_obj <- get(spec$dataset)
  mesh_obj <- get(spec$mesh_name)
  
  n_rows <- nrow(data_obj)
  n_mesh_vertices <- tryCatch(mesh_obj$mesh$n, error = function(e) NA)
  
  start_time <- Sys.time()
  
  mod <- tryCatch({
    sdmTMB(
      formula = as.formula(size_formula),
      spatial = "on",
      mesh = mesh_obj,
      family = tweedie(link = "log"),
      data = data_obj
    )
  }, error = function(e) {
    message("Model failed: ", spec$name, " -- ", conditionMessage(e))
    NULL
  })
  
  end_time <- Sys.time()
  run_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
  
  if (!is.null(mod)) {
    #saveRDS(mod, file = file.path("output_models", paste0(spec$name, ".rds")))
    size_models[[spec$name]] <- mod
    
    coefs <- tryCatch(tidy(mod, effects = "fixed", conf.int = TRUE), error = function(e) NULL)
    coefs_ran <- tryCatch(tidy(mod, effects = "ran_pars", conf.int = TRUE), error = function(e) NULL)
    
    if (!is.null(coefs)) coefs$name <- spec$name
    if (!is.null(coefs_ran)) coefs_ran$name <- spec$name
    
    size_coef_list[[spec$name]] <- list(fixed = coefs, ran_pars = coefs_ran)
    
    converged <- !is.null(mod$sd_report) && mod$sd_report$pdHess
    max_gradient <- tryCatch(max(abs(mod$gradients)), error = function(e) NA)
    aic_val <- tryCatch(AIC(mod), error = function(e) NA)
  } else {
    converged <- NA
    max_gradient <- NA
    aic_val <- NA
  }
  
  size_results <- rbind(size_results, data.frame(
    name = spec$name,
    dataset = spec$dataset,
    n_locations = spec$n_locations,
    n_rows = n_rows,
    n_mesh_vertices = n_mesh_vertices,
    run_time_secs = run_time,
    converged = converged,
    max_gradient = max_gradient,
    AIC = aic_val
  ))
}

#write.csv(size_results, "output_models/size_comparison.csv", row.names = FALSE)

fixed_coefs_all <- do.call(rbind, lapply(size_coef_list, function(x) x$fixed))
#write.csv(fixed_coefs_all, "output_models/size_coefficients.csv", row.names = FALSE)

size_results

size_results$size_label <- factor(size_results$n_locations,
                                  levels = c(100, 500, 1000, 5000),
                                  labels = c("100", "500", "1000", "5000"))

ggplot(size_results, aes(x = size_label, y = run_time_secs)) +
  geom_col(fill = "steelblue") +
  labs(x = "Number of locations", y = "Run time (seconds)",
       title = "Model run time by dataset size (mesh cutoff = 5)") +
  theme_bw()

ggplot(size_results, aes(x = n_locations, y = run_time_secs)) +
  geom_point(size = 3, color = "steelblue") +
  geom_line(color = "steelblue") +
  labs(x = "Number of locations", y = "Run time (seconds)",
       title = "Model run time scaling with dataset size") +
  theme_bw()

ggplot(size_results, aes(x = n_rows, y = run_time_secs)) +
  geom_point(size = 3, color = "steelblue") +
  geom_line(color = "steelblue") +
  labs(x = "Number of rows in dataframe", y = "Run time (seconds)",
       title = "Model run time scaling with number of rows") +
  theme_bw()




# Compare different basis dimensions k for the smoothers
k_specs <- data.frame(
  name = c("k4", "k6", "k8", "k10"),
  k = c(4, 6, 8, 10),
  stringsAsFactors = FALSE
)

k_results <- data.frame()
k_models <- list()
k_coef_list <- list()
pred_pr_list <- list()      # store predicted-response data per k, for pr_sum_sc
pred_prcv_list <- list()    # store predicted-response data per k, for pr_sum_cv_sc
resid_pr_list <- list()     # store residual data per k, for pr_sum_sc
resid_prcv_list <- list()   # store residual data per k, for pr_sum_cv_sc

for (i in seq_len(nrow(k_specs))) {
  spec <- k_specs[i, ]
  cat("Fitting:", spec$name, "\n")
  
  form <- as.formula(paste0(
    "Npp ~ poly(tmmx_mean_sc, 2) + s(pr_sum_sc, bs = 'cr', k = ", spec$k,
    ") + s(pr_sum_cv_sc, bs = 'cr', k = ", spec$k,
    ") + elevation_mean_sc + year_sc"
  ))
  
  start_time <- Sys.time()
  
  mod <- tryCatch({
    sdmTMB(
      formula = form,
      spatial = "on",
      mesh = mesh_500_5,
      family = tweedie(link = "log"),
      data = df_500
    )
  }, error = function(e) {
    message("Model failed: ", spec$name, " -- ", conditionMessage(e))
    NULL
  })
  
  end_time <- Sys.time()
  run_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
  
  if (!is.null(mod)) {
    saveRDS(mod, file = file.path("output_models", paste0(spec$name, ".rds")))
    k_models[[spec$name]] <- mod
    
    coefs <- tryCatch(tidy(mod, effects = "fixed", conf.int = TRUE), error = function(e) NULL)
    coefs_ran <- tryCatch(tidy(mod, effects = "ran_pars", conf.int = TRUE), error = function(e) NULL)
    
    if (!is.null(coefs)) coefs$name <- spec$name
    if (!is.null(coefs_ran)) coefs_ran$name <- spec$name
    
    k_coef_list[[spec$name]] <- list(fixed = coefs, ran_pars = coefs_ran)
    
    converged <- !is.null(mod$sd_report) && mod$sd_report$pdHess
    max_gradient <- tryCatch(max(abs(mod$gradients)), error = function(e) NA)
    aic_val <- tryCatch(AIC(mod), error = function(e) NA)
    
    # ---- Predicted response, collected for combined overlay plots ----
    pred_pr <- as.data.frame(predict_response(mod, terms = "pr_sum_sc [all]", ci_level = NA))
    pred_pr$k <- factor(spec$k)
    pred_pr_list[[spec$name]] <- pred_pr
    
    pred_prcv <- as.data.frame(predict_response(mod, terms = "pr_sum_cv_sc [all]", ci_level = NA))
    pred_prcv$k <- factor(spec$k)
    pred_prcv_list[[spec$name]] <- pred_prcv
    
    # ---- Residuals, collected for combined overlay plots ----
    resid_mod <- tryCatch(residuals(mod, type = "mle-mvn"), error = function(e) NULL)
    
    if (!is.null(resid_mod)) {
      resid_pr_list[[spec$name]] <- data.frame(
        covariate = mod$data$pr_sum_sc, resid = resid_mod, k = factor(spec$k)
      )
      resid_prcv_list[[spec$name]] <- data.frame(
        covariate = mod$data$pr_sum_cv_sc, resid = resid_mod, k = factor(spec$k)
      )
    }
    
  } else {
    converged <- NA
    max_gradient <- NA
    aic_val <- NA
  }
  
  k_results <- rbind(k_results, data.frame(
    name = spec$name,
    k = spec$k,
    run_time_secs = run_time,
    converged = converged,
    max_gradient = max_gradient,
    AIC = aic_val
  ))
}

write.csv(k_results, "output_models/k_comparison.csv", row.names = FALSE)

fixed_coefs_all <- do.call(rbind, lapply(k_coef_list, function(x) x$fixed))
ran_coefs_all <- do.call(rbind, lapply(k_coef_list, function(x) x$ran_pars))
write.csv(fixed_coefs_all, "output_models/k_fixed_coefficients.csv", row.names = FALSE)
write.csv(ran_coefs_all, "output_models/k_ran_coefficients.csv", row.names = FALSE)

k_results

pred_pr_all <- do.call(rbind, pred_pr_list)
pred_prcv_all <- do.call(rbind, pred_prcv_list)

p_pred_pr <- ggplot(pred_pr_all, aes(x = x, y = predicted, color = k)) +
  geom_line(linewidth = 1) +
  labs(x = "pr_sum_sc", y = "Predicted response", color = "k",
       title = "Predicted response - pr_sum_sc, by k") +
  theme_bw()
ggsave("output_models/pred_pr_by_k.png", p_pred_pr, width = 7, height = 5)
p_pred_pr

p_pred_prcv <- ggplot(pred_prcv_all, aes(x = x, y = predicted, color = k)) +
  geom_line(linewidth = 1) +
  labs(x = "pr_sum_cv_sc", y = "Predicted response", color = "k",
       title = "Predicted response - pr_sum_cv_sc, by k") +
  theme_bw()
ggsave("output_models/pred_pr_cv_by_k.png", p_pred_prcv, width = 7, height = 5)
p_pred_prcv

resid_pr_all <- do.call(rbind, resid_pr_list)
resid_prcv_all <- do.call(rbind, resid_prcv_list)

p_resid_pr <- ggplot(resid_pr_all, aes(x = covariate, y = resid, color = k)) +
  geom_smooth(method = "gam", formula = y ~ s(x, bs = "cs"), se = FALSE, linewidth = 1) +
  geom_hline(yintercept = 0, color = "black", linetype = "dashed") +
  labs(x = "pr_sum_sc", y = "Randomized quantile residuals", color = "k",
       title = "Residuals vs. pr_sum_sc, by k") +
  theme_bw()
ggsave("output_models/resid_pr_by_k.png", p_resid_pr, width = 7, height = 5)
p_resid_pr

p_resid_prcv <- ggplot(resid_prcv_all, aes(x = covariate, y = resid, color = k)) +
  geom_smooth(method = "gam", formula = y ~ s(x, bs = "cs"), se = FALSE, linewidth = 1) +
  geom_hline(yintercept = 0, color = "black", linetype = "dashed") +
  labs(x = "pr_sum_cv_sc", y = "Randomized quantile residuals", color = "k",
       title = "Residuals vs. pr_sum_cv_sc, by k") +
  theme_bw()
ggsave("output_models/resid_pr_cv_by_k.png", p_resid_prcv, width = 7, height = 5)
p_resid_prcv

k_results$k_label <- factor(k_results$k, levels = c(4, 6, 8, 10))

p_runtime_bar <- ggplot(k_results, aes(x = k_label, y = run_time_secs)) +
  geom_col(fill = "steelblue") +
  labs(x = "k (basis dimension)", y = "Run time (seconds)",
       title = "Model run time by smoother k value") +
  theme_bw()
ggsave("output_models/runtime_by_k.png", p_runtime_bar, width = 6, height = 4)
p_runtime_bar

p_runtime_line <- ggplot(k_results, aes(x = k, y = run_time_secs)) +
  geom_point(size = 3, color = "steelblue") +
  geom_line(color = "steelblue") +
  labs(x = "k (basis dimension)", y = "Run time (seconds)",
       title = "Model run time scaling with k") +
  theme_bw()
ggsave("output_models/runtime_by_k_line.png", p_runtime_line, width = 6, height = 4)
p_runtime_line

#* Subsection 6.3: Compare results of diff. random samples of 5000 locs ----

# Create random samples of 5000 locations
take_disjoint_samples <- function(df, n_locs, n_samples, seed = 123) {
  set.seed(seed)
  
  all_locs <- unique(df$location_id)
  
  if (length(all_locs) < n_locs * n_samples) {
    stop("Not enough unique locations (", length(all_locs), ") to draw ",
         n_samples, " disjoint samples of ", n_locs, " each (need ",
         n_locs * n_samples, ").")
  }
  
  # Shuffle once, then slice into n_samples non-overlapping chunks
  shuffled_locs <- sample(all_locs)
  
  samples <- vector("list", n_samples)
  for (i in seq_len(n_samples)) {
    idx <- ((i - 1) * n_locs + 1):(i * n_locs)
    locs_i <- shuffled_locs[idx]
    df_i <- df[df$location_id %in% locs_i, ]
    samples[[i]] <- df_i
  }
  
  names(samples) <- letters[1:n_samples]
  return(samples)
}

# Draw 5 disjoint samples of 5000 locations each, all at once
samples_5000 <- take_disjoint_samples(df, n_locs = 5000, n_samples = 5, seed = 123)

df_5000a <- samples_5000$a
df_5000b <- samples_5000$b
df_5000c <- samples_5000$c
df_5000d <- samples_5000$d
df_5000e <- samples_5000$e

# Confirm sizes and no overlap
sapply(samples_5000, function(d) length(unique(d$location_id)))

all_locs_combined <- unlist(lapply(samples_5000, function(d) unique(d$location_id)))
length(all_locs_combined) == length(unique(all_locs_combined))  # should be TRUE


# Create meshes
mesh_5000a_5 <- create_mesh(df_5000a, 5)
mesh_5000b_5 <- create_mesh(df_5000b, 5)
mesh_5000c_5 <- create_mesh(df_5000c, 5)
mesh_5000d_5 <- create_mesh(df_5000d, 5)
mesh_5000e_5 <- create_mesh(df_5000e, 5)


# Run model on different samples
sample_specs <- data.frame(
  name = c("sample_a", "sample_b", "sample_c", "sample_d", "sample_e"),
  dataset = c("df_5000a", "df_5000b", "df_5000c", "df_5000d", "df_5000e"),
  mesh_name = c("mesh_5000a_5", "mesh_5000b_5", "mesh_5000c_5", "mesh_5000d_5", "mesh_5000e_5"),
  stringsAsFactors = FALSE
)

sample_formula <- "Npp ~ poly(tmmx_mean_sc, 2) + s(pr_sum_sc) + s(pr_sum_cv_sc) + elevation_mean_sc + year_sc"

sample_results <- data.frame()
sample_models <- list()
sample_coef_list <- list()

for (i in seq_len(nrow(sample_specs))) {
  spec <- sample_specs[i, ]
  cat("Fitting:", spec$name, "\n")
  
  data_obj <- get(spec$dataset)
  mesh_obj <- get(spec$mesh_name)
  
  start_time <- Sys.time()
  
  mod <- tryCatch({
    sdmTMB(
      formula = as.formula(sample_formula),
      spatial = "on",
      mesh = mesh_obj,
      family = tweedie(link = "log"),
      data = data_obj
    )
  }, error = function(e) {
    message("Model failed: ", spec$name, " -- ", conditionMessage(e))
    NULL
  })
  
  end_time <- Sys.time()
  run_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
  
  if (!is.null(mod)) {
    saveRDS(mod, file = file.path("output_models", paste0(spec$name, ".rds")))
    sample_models[[spec$name]] <- mod
    
    coefs <- tryCatch(tidy(mod, effects = "fixed", conf.int = TRUE), error = function(e) NULL)
    coefs_ran <- tryCatch(tidy(mod, effects = "ran_pars", conf.int = TRUE), error = function(e) NULL)
    
    if (!is.null(coefs)) coefs$name <- spec$name
    if (!is.null(coefs_ran)) coefs_ran$name <- spec$name
    
    sample_coef_list[[spec$name]] <- list(fixed = coefs, ran_pars = coefs_ran)
    
    converged <- !is.null(mod$sd_report) && mod$sd_report$pdHess
    max_gradient <- tryCatch(max(abs(mod$gradients)), error = function(e) NA)
    aic_val <- tryCatch(AIC(mod), error = function(e) NA)
  } else {
    converged <- NA
    max_gradient <- NA
    aic_val <- NA
  }
  
  sample_results <- rbind(sample_results, data.frame(
    name = spec$name,
    dataset = spec$dataset,
    run_time_secs = run_time,
    converged = converged,
    max_gradient = max_gradient,
    AIC = aic_val
  ))
}

write.csv(sample_results, "output_models/5000_random_sample_comparison.csv", row.names = FALSE)

fixed_coefs_all <- do.call(rbind, lapply(sample_coef_list, function(x) x$fixed))
write.csv(fixed_coefs_all, "output_models/5000_random_sample_fixed_coefs.csv", row.names = FALSE)
ran_coefs_all <- do.call(rbind, lapply(sample_coef_list, function(x) x$ran_pars))
write.csv(ran_coefs_all, "output_models/5000_random_sample_ran_coefs.csv", row.names = FALSE)

sample_results
fixed_coefs_all
ran_coefs_all

# Plot differences in fixed coefficients across samples
p_fixed <- ggplot(fixed_coefs_all, aes(x = name, y = estimate, color = name)) +
  geom_pointrange(aes(ymin = conf.low, ymax = conf.high), size = 0.6) +
  facet_wrap(~ term, scales = "free_y") +
  labs(x = "Sample", y = "Estimate (95% CI)", color = "Sample",
       title = "Fixed effect coefficients across random samples") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "none")

ggsave("output_models/5000_random_sample_fixed_coefs.png", p_fixed, width = 8, height = 6)
p_fixed

# Plot differences in range/variance coefficients across samples
p_ran <- ggplot(ran_coefs_all, aes(x = name, y = estimate, color = name)) +
  geom_pointrange(aes(ymin = conf.low, ymax = conf.high), size = 0.6) +
  facet_wrap(~ term, scales = "free_y") +
  labs(x = "Sample", y = "Estimate (95% CI)", color = "Sample",
       title = "Random/variance parameters by sample") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "none")

ggsave("output_models/5000_random_sample_ran_coefs.png", p_ran, width = 8, height = 6)
p_ran


# Section 7: Model interpretation ----


# Look at an example model
# mod2 <- readRDS("output_models/yearly_model_smoothtemp-smoothpr-smoothcv_space_time.rds")
# mod2 <- readRDS("output_models/sample_d.rds")
  
# Coefficients
tidy(mod2, conf.int = TRUE)

# Parameters
tidy(mod2, effects = "ran_pars", conf.int = TRUE)
#'The Tweedie dispersion (phi) and power parameters control the distribution’s 
#'mean-variance relationship. The Matérn range is the distance at which spatial 
#'correlation becomes negligible (~0.13 correlation). The marginal spatial field 
#'standard deviation (sigma_O) represents unexplained spatial variation.

# Basic sanity check
sanity(mod2)

# Plot fitted values
pred <- predict_response(mod4, terms = "pr_sum_sc [all]", ci_level = NA) #[-2:4]
plot(pred)

# Apply the model's link function manually to transform back to the link scale to verify that the modelled relation is linear
link_fun <- family(mod2)$linkfun
link_fun
pred$predicted <- link_fun(pred$predicted)
pred$conf.low <- link_fun(pred$conf.low)
pred$conf.high <- link_fun(pred$conf.high)
plot(pred)

#' intercept has very large confidence intervals
#' TODO: choose how to marginalize over non-focal predictors with the "margin" argument

pred1 <- predict_response(mod4, terms = "tmmx_mean_sc [all]", ci_level = NA) # include [all] in the terms string to get a smooth plot
plot(pred1)

pred2 <- predict_response(mod4, terms = "pr_sum_cv_sc [all]", ci_level = NA)
plot(pred2)

pred3 <- predict_response(mod4, terms = "elevation_mean_sc [all]", ci_level = NA)
plot(pred3)


# Section 8: Model validation ----

#* Subsection 8.0: Extract residuals ----

#' I use two types of residuals:
#'
#'   1. Analytical randomized-quantile residuals - normal distribution
#'   2. Simulation-based randomized-quantile residuals from DHARMa - uniform distribution

# Extract analytical randomized-quantile residuals
r1 <- residuals(mod2, type = "mle-mvn")

# Extract simulation-based randomized-quantile residuals from DHARMa
sim <- simulate(mod2, nsim = 500, type = "mle-mvn") # start 11:32
r2 <- dharma_residuals(sim, mod2, return_DHARMa = TRUE)

# Extract simulation-based randomized-quantile residuals from DHARMa
sim <- simulate(mod4, nsim = 500, type = "mle-mvn")
r3 <- dharma_residuals(sim, mod4, return_DHARMa = TRUE)

#'The randomized quantile residuals in residuals.sdmTMB() are returned such that 
#'they will be normal(0, 1) if the model is consistent with the data. DHARMa residuals, 
#'however, are returned as uniform(0, 1) under those same circumstances. 

sim_2002 <- simulate(mod2_2002, nsim = 500, type = "mle-mvn")
r2_2002 <- dharma_residuals(sim_2002, mod2_2002, return_DHARMa = TRUE)

#* Subsection 8.1: Residual distribution ----

# Analytical
hist(r1)
qqnorm(r1);abline(a=0, b=1)

# DHARMa
hist(r2)
plotQQunif(r2)
# general tests
testResiduals(r2)


#* Subsection 8.2: Residuals vs. fitted values ---- 

# Analytical
plot(r1)

fitted_vals_sdm <- fitted(mod2)
df_resid <- data.frame(fitted = fitted_vals_sdm, resid = r1)
ggplot(df_resid, aes(x = fitted, y = resid)) +
  geom_smooth(method = "gam", formula = y ~ s(x, bs = "cs"), color = "steelblue", linewidth = 1) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed", linewidth = 1) +
  labs(x = "Fitted values", y = "Randomized quantile residuals",
       title = "Residuals vs. Fitted (smoother) - sdmTMB model") +
  theme_bw()

# DHARMa
plotResiduals(r2)

#' too large fitted values for large Npp i think

#* Subsection 8.3: Residuals vs. covariates (in and not in the model) ----

# Analytical
covariates <- c("tmmx_mean_sc", "pr_sum_sc", "pr_sum_cv_sc", "elevation_mean_sc", "vegetation_length_sc")

plots <- list()

for (cov in covariates) {
  df_resid_cov <- data.frame(
    covariate = mod2$data[[cov]],
    resid = r1
  )
  
  p <- ggplot(df_resid_cov, aes(x = covariate, y = resid)) +
    geom_smooth(method = "gam", formula = y ~ s(x, bs = "cs"), color = "steelblue", linewidth = 1) +
    geom_hline(yintercept = 0, color = "red", linetype = "dashed", linewidth = 1) +
    labs(x = cov, y = "Randomized quantile residuals",
         title = paste("Residuals vs.", cov)) +
    theme_bw()
  
  plots[[cov]] <- p
  print(p)
}
plot(r1 ~ df_sample$tmmx_mean_sc)

# DHARMa: set rank = FALSE??
plotResiduals(r2, form = mod2$data$tmmx_mean_sc)
plotResiduals(r2, form = mod2$data$pr_sum_sc)
plotResiduals(r2, form = mod2$data$pr_sum_cv_sc)
plotResiduals(r2, form = mod2$data$elevation_mean_sc)
plotResiduals(r2, form = mod2$data$vegetation_length_sc)


#* Subsection 8.4: Overdispersion and zero-inflation ----

# Analytical
var(r1)  # should be close to 1 if dispersion is well-calibrated
sd(r1)   # should be close to 1
         # does not have a direct overdispersion and zeroinflation test

# DHARMa
testDispersion(r2)
testZeroInflation(r2)


#* Subsection 8.5: Residuals vs. time ----

# Analytical
df_resid_year <- data.frame(
  year_sc = mod2$data$year_sc,
  resid = r1
)

ggplot(df_resid_year, aes(x = year_sc, y = resid)) +
  geom_smooth(method = "gam", formula = y ~ s(x, bs = "cs"), color = "steelblue", linewidth = 1) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed", linewidth = 1) +
  labs(x = "year_sc", y = "Randomized quantile residuals",
       title = "Residuals vs. year_sc - sdmTMB model") +
  theme_bw()

# DHARMa
plotResiduals(r2, form = mod$data$year_sc)


# Autocorrelation
set.seed(123)
n_sample_acf <- 20  # how many locations to check
locations_acf_sample <- sample(unique(df_sample$location_id), size = n_sample_acf)

# Attach residuals to df_sample for easy subsetting (safe since r2 was computed directly on df_sample)
df_sample$resid <- r3$scaledResiduals # replace with r1 for analytical residuals 

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


# test for 1 year
testSpatialAutocorrelation(r2_2002, x = df_sample_2002$longitude, y = df_sample_2002$latitude)


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


# variogram for 1 year
# Combine residuals with coordinates for this single year
resid_df_2002 <- data.frame(
  longitude = df_sample_2002$longitude,
  latitude = df_sample_2002$latitude,
  residual = r2_2002$scaledResiduals
)

# Convert to a spatial object
coordinates(resid_df_2002) <- ~ longitude + latitude

# Compute and plot the empirical variogram
vg_2002 <- variogram(residual ~ 1, data = resid_df_2002)
plot(vg_2002, main = "Variogram of residuals - 2002")




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

fitted_vals <- fitted(mod4)
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


# Linear model vegetation period
start_time <- Sys.time()

modveg1 <- glmmTMB(Npp ~ veg_tmmx_mean_sc + veg_pr_sum_sc + veg_pr_sum_cv_sc + vegetation_length_sc + elevation_mean_sc, 
                   family = tweedie(link = "log"), REML = TRUE, data = df_sample)

end_time <- Sys.time()
run_time <- end_time - start_time
print(run_time)

summary(modveg1)
family_params(modveg1)

# Linear model yearly with time
start_time <- Sys.time()

mod2 <- glmmTMB(Npp ~ tmmx_mean_sc + pr_sum_sc + pr_sum_cv_sc + elevation_mean_sc + year_sc, 
                family = tweedie(link = "log"), REML = TRUE, data = df_sample)

end_time <- Sys.time()
run_time <- end_time - start_time
print(run_time)

summary(mod2)
family_params(mod2)



#' TODO:  
#' 
#' later
#' - NAs in vegmodels! maybe exclude from the start?
#' - modelling: adjust spatial mesh
#' - how to model: we expect that only for some pixels (degraded ones) the Npp stays stable although precipitation increases
#' - convert lon/lat to better crs that preserves distances
#' - does the spatial correlation change over time? whats the difference between including year as covariate or time component?
#' - do we have spatial or time varying covariates?
#' - dharma plotResiduals: set rank = FALSE??
#' - choose how to marginalize over non-focal predictors with the "margin" argument for plotting effects of predictors / fitted values
#' - check if any variable leads to higher variance in Npp
#' - choose exact parameters for the smooth terms


# Trash ----

#* Try to create a mesh based on great circle distances ----
library(sf)
library(fmesher)

# 1. Convert your coordinates into an sf object with the WGS84 CRS (EPSG 4326)
# (Assuming your data frame is called 'my_data' with columns 'lon' and 'lat')
points_sf <- st_as_sf(df_sample, coords = c("longitude", "latitude"), crs = 4326)
points_sf
# 2. Build the mesh using the sf object.
# Adding a bounding globe allows fmesher to realize it should embed it on a sphere.
mesh <- fm_mesh_2d(
  loc = points_sf,
  #globe = 1,         # Keeps it on a unit sphere manifold
  cutoff = 5,
  manifold = "S2"# Controls mesh resolution
)

# 3. Verify the manifold
mesh
plot(mesh)
# Should now return: "S2"

