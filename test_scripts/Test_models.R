# ---- Test script for trying models ----

# ---- Load data ----

# install.packages("arrow")
#install.packages("statmod")
#install.packages("tweedie")
# install.packages("peakRAM")
#install.packages("fmesher")
# Warning: dependencies ‘Rgraphviz’, ‘graph’ are not available
# also installing the dependencies ‘bayesm’, ‘foreach’, ‘iterators’, ‘spam’, ‘maps’, ‘litedown’, ‘dfidx’, ‘micsr’,
#‘coda’, ‘spData’, ‘INLAtools’, ‘inlabru’, ‘compositions’, ‘Ecdat’, ‘deldir’, ‘doParallel’, ‘evd’, ‘fastGHQuad’,
#‘fields’, ‘gsl’, ‘markdown’, ‘matrixStats’, ‘mlogit’, ‘pixmap’, ‘rgl’, ‘runjags’, ‘sn’, ‘spdep’, ‘tidyterra’, ‘INLAspacetime’
# BiocManager::install(c("graph", "Rgraphviz"), dep=TRUE)
# install.packages("INLA", repos=c(getOption("repos"), INLA="https://inla.r-inla-download.org/R/stable"), dep=FALSE)

# Load libraries
library(arrow)     # for loading parquet
library(dplyr)
library(statmod)   # for tweedie() family
library(tweedie)   # for estimating the power parameter
library(INLA)
library(corrplot)
library(nlme)


# Set working directory
setwd("/Users/Wanja/Documents/non-equilibrium_data")

# Load the file
df <- read_parquet("cv_new/table_wgs84_2002_with_cv.parquet")
df_years <- read_parquet("cv_new/table_wgs84_2002-2018_with_cv_sample.parquet")

# Check it loaded correctly
head(df)            # first rows
str(df)             # columns
dim(df)             # number of rows/columns
sapply(df, class)   # variable types — check factors vs numeric are as expected
sum(is.na(df))      # missing values overall
colSums(is.na(df))  # missing values by column


# ---- Check distributions ----

hist(df$Npp)
hist(sqrt(df$Npp))
summary(df$Npp)
sum(df$Npp == 0)


# ---- Check multicollinearity ----

# Select all numeric columns, excluding Npp and year
df_num_full <- df %>%
  select(where(is.numeric)) %>%
  select(-Npp, -year, -Npp_QC, -Gpp, -Npp_cv)

# Random sample of 100,000 rows
set.seed(123)
df_num <- df_num_full[sample(nrow(df_num_full), 100000), ]

dim(df_num)  # sanity check

# Spearman correlation matrix
cor_matrix <- cor(df_num_full, method = "spearman", use = "complete.obs")
print(cor_matrix)

corrplot(cor_matrix,
         method = "color",       # colored squares (other options: "circle", "number", "shade")
         type = "upper",         # only show upper triangle (avoids redundant mirror)
         order = "hclust",       # cluster similar variables together — helpful for spotting groups
         tl.col = "black",       # text label color
         tl.srt = 45,            # rotate variable labels for readability
         diag = FALSE,           # hide the diagonal (all 1s, not informative)
         addCoef.col = "black",  # overlay the actual correlation values
         number.cex = 0.7)       # size of the coefficient text

# ---- Check VIF ----

#library(car)
# Fit a placeholder linear model with all predictors to check VIF
#vif_mod <- lm(Npp ~ ., data = df %>% select(where(is.numeric)) %>% select(-year))
#vif(vif_mod)


# ---- Try GLM ----

#mod_pois <- glm(Npp ~ pr_sum, data = df, family = poisson(link = "log"))
#summary(mod_pois)

mod_linear <- lm(sqrt(Npp) ~ pr_sum + pr_sum_cv + tmmn_mean + elevation_mean, data = df)
summary(mod_linear)

mod_linear_years <- lm(sqrt(Npp) ~ pr_sum + pr_sum_cv + tmmn_mean + elevation_mean, data = df_years)
summary(mod_linear_years)


# ---- Try a LMM ----

# normal mixed effect model runs on 4 mio points
mod_mixed <- lme(sqrt(Npp) ~ pr_sum + pr_sum_cv + tmmn_mean + elevation_mean,
                 random = ~ 1 | continent,
                 data = df,
                 na.action = na.omit)

summary(mod_mixed) # looks good, but inflated p-values due to pseudoreplication
hist(mod_mixed$residuals) # residuals are normally distributed
# plot(mod_mixed) on a small sample the variance was larger for larger fitted values

# mixed effect model over the years runs on 1.7 mio points
mod_mixed_years <- lme(sqrt(Npp) ~ pr_sum + pr_sum_cv + tmmn_mean + elevation_mean,
                 random = ~ year | continent,
                 data = df_years,
                 na.action = na.omit)
summary(mod_mixed_years)
hist(mod_mixed_years$residuals) # very weird residuals!


# ---- Try Bayesian GLM ----

# r-inla

# sample random points
# 1 mio rows works for lmm; 2 mio rows still works for a simple model; 4 mio crashed
# adding 1 variable seems to make a bit of a difference (50 vs 70% Auslastung bei 2 mio rows)
set.seed(123)  # for reproducibility
df_sample <- df[sample(nrow(df), 3000000), ]
dim(df_sample)

# run this command in the terminal while running r-inla to check the RAM used: top -o mem
# reducing num.threads reduces the Auslastung, increases runtime
#mod_inla <- inla(sqrt(Npp) ~ pr_sum + pr_sum_cv + tmmn_mean + elevation_mean, data = df_sample, family = "gaussian")
mod_inla

lmm_inla <- inla(sqrt(Npp) ~ pr_sum + pr_sum_cv + tmmn_mean + elevation_mean + f(continent, model = "iid"),
                 data = df_sample, family = "gaussian", num.threads=1)
lmm_inla


# ---- Bayesian GLM with RAM monitoring ----
library(callr)
library(ps)

get_named_process_rss_gb <- function(pattern = "inla") {
  procs <- ps::ps()
  matches <- procs[grepl(pattern, procs$name, ignore.case = TRUE), ]
  if (nrow(matches) == 0) return(0)
  total <- sum(sapply(matches$pid, function(p) {
    tryCatch(ps::ps_memory_info(ps::ps_handle(p))["rss"], error = function(e) 0)
  }))
  total / 1e9
}

rx <- r_bg(function(df_sample) {
  library(INLA)
  inla(sqrt(Npp) ~ pr_sum + pr_sum_cv + tmmn_mean + elevation_mean +
         f(continent, model = "iid"),
       data = df_sample, family = "gaussian", num.threads = 2)
}, args = list(df_sample = df_sample))

limit_gb <- 15   # <- set this to whatever ceiling makes sense for your Mac
poll_every <- 0.5   # seconds

peak_gb <- 0
while (rx$is_alive()) {
  Sys.sleep(poll_every)
  mem_gb  <- get_named_process_rss_gb()
  peak_gb <- max(peak_gb, mem_gb)
  if (mem_gb > limit_gb) {
    rx$kill_tree()
    stop("Killed: process tree exceeded ", limit_gb, " GB (was ", round(mem_gb, 1), ")")
  }
}

message("Peak memory observed: ", round(peak_gb, 2), " GB")

lmm_inla <- rx$get_result()
summary(lmm_inla)

# use the Aktivitätsanzeige to track CPU & Memory/Speicher, that one seems accurate
# the tracker in R now check the inla process, but always detects a bit later/lower memory, so use a conservative threshold to kill processes
# my Mac has physical memory of 18GB, but it can in addition store sth on SSD which it does via the "Swap"
# I can thus run processes that need more RAM than my laptop has, but idk how much more
# num.threads: reducing num.threads reduces the Auslastung and also memory a bit (bc multiple threads store it multiple times or sth), but increases runtime

# 3 mio rows sample Bayesian LMM still worked but did have high memory usage (16 physical, + 17 swap or sth)

# ---- Try Tweedie GLM ----

# Step 1: estimate the power parameter (p) via profile likelihood
out <- tweedie.profile(Npp ~ pr_sum, data = df,
                       p.vec = seq(1.1, 1.9, by = 0.1))
p_est <- out$p.max

# Step 2: fit the GLM with the estimated power parameter
mod <- glm(Npp ~ pr_sum, data = df,
           family = tweedie(var.power = p_est, link.power = 0))

summary(mod)
