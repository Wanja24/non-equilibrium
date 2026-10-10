# Model validation script - overnight

# Section 7 & 8: Model interpretation & validation ----

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

# Returns a function converting a scaled covariate back to its original units.
# Works because x_sc = (x - mean) / sd is exactly linear, so mean and sd can be
# recovered from the model data, provided the unscaled column is in mod$data.
make_unscaler <- function(sc_name, data) {
  orig_name <- sub("_sc$", "", sc_name)
  if (orig_name %in% names(data)) {
    cf <- coef(lm(data[[orig_name]] ~ data[[sc_name]]))
    return(function(x) cf[[1]] + cf[[2]] * x)
  }
  if (sc_name == "year_sc") return(function(x) x + 2000)  # year_sc = year - 2000
  NULL
}

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
  model_vars <- setdiff(all.vars(mod$formula[[1]]), "Npp")
  pred_list <- list()
  
  for (term in model_vars) {
    pred <- tryCatch(predict_response(mod, terms = paste0(term, " [all]"), ci_level = NA),
                     error = function(e) NULL)
    if (!is.null(pred)) {
      df_pred <- as.data.frame(pred)
      df_pred$x_sc <- df_pred$x
      
      unscale <- make_unscaler(term, mod$data)
      if (!is.null(unscale)) df_pred$x <- unscale(df_pred$x_sc)
      x_label <- if (!is.null(unscale)) sub("_sc$", "", term) else term
      
      pred_list[[term]] <- df_pred
      p <- ggplot(df_pred, aes(x = x, y = predicted)) +
        geom_line(linewidth = 1) +
        labs(x = x_label, y = "Predicted Npp") +
        theme_plain()
      ggsave(file.path(diag_folder, paste0("pred_", term, ".png")), p,
             width = gg_width, height = gg_height, dpi = 300)
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
    png_plot(file.path(diag_folder, "r2_histogram.png"))
    hist(r2)
    dev.off()
    
    png_plot(file.path(diag_folder, "r2_qq_uniform.png"))
    plotQQunif(r2)
    dev.off()
    
    # Residuals vs fitted
    png_plot(file.path(diag_folder, "r2_resid_vs_fitted.png"))
    plotResiduals(r2)
    dev.off()
    
    # Residuals vs each covariate
    for (cov in model_vars) {
      if (cov %in% names(mod$data)) {
        png_plot(file.path(diag_folder, paste0("r2_resid_vs_", cov, ".png")))
        tryCatch(plotResiduals(r2, form = mod$data[[cov]], xlab = cov), error = function(e) NULL)
        dev.off()
      }
    }
    
    # Dispersion and zero-inflation plots (plots only, test results not saved)
    png_plot(file.path(diag_folder, "r2_dispersion_test.png"))
    tryCatch(testDispersion(r2), error = function(e) NULL)
    dev.off()
    
    png_plot(file.path(diag_folder, "r2_zeroinflation_test.png"))
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
    png_plot(file.path(diag_folder, "acf_hist_by_lag.png"), width = 900, height = 600)
    par(mfrow = c(2, 3))
    for (i in 1:5) {
      hist(acf_resid_by_loc[, i], main = paste("Lag", i), xlab = "ACF")
      abline(v = 0, col = "red", lty = 2)
    }
    par(mfrow = c(1, 1))
    dev.off()
    
    # Mean ACF decay curve
    png_plot(file.path(diag_folder, "acf_mean_decay.png"))
    plot(1:5, colMeans(acf_resid_by_loc, na.rm = TRUE), type = "b", pch = 16,
         xlab = "Lag", ylab = "Mean residual autocorrelation",
         main = "Average temporal ACF decay of residuals")
    abline(h = 0, col = "red", lty = 2)
    dev.off()
    
    # Boxplot per lag
    png_plot(file.path(diag_folder, "acf_boxplot_by_lag.png"))
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
    
    png_plot(file.path(diag_folder, "spatial_variograms_by_year.png"),
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
  }
  
  # ---- Observed vs. fitted ----
  fitted_vals <- tryCatch(predict(mod, type = "response")$est, error = function(e) NULL)
  
  if (!is.null(fitted_vals)) {
    df_plot <- data.frame(fitted = fitted_vals, observed = mod$data$Npp)
    
    # Scatter (semi-transparent points)
    png_plot(file.path(diag_folder, "obs_vs_fitted_scatter.png"))
    plot(df_plot$fitted, df_plot$observed,
         xlab = "Fitted values", ylab = "Observed values",
         main = NULL,
         pch = 16, col = rgb(0, 0, 0, 0.3))
    abline(0, 1, col = "red", lwd = 2, lty = 2)
    dev.off()
    
    # Hexbin
    p_hex <- ggplot(df_plot, aes(x = fitted, y = observed)) +
      geom_hex(bins = 50) +
      geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed", linewidth = 1) +
      scale_fill_viridis_c() +
      labs(x = "Fitted values", y = "Observed values", fill = "Count") +
      theme_plain()
    ggsave(file.path(diag_folder, "obs_vs_fitted_hex.png"), p_hex, width = gg_width, height = gg_height, dpi = 300)
    
    # Smoother
    p_smooth <- ggplot(df_plot, aes(x = fitted, y = observed)) +
      geom_smooth(method = "gam", formula = y ~ s(x, bs = "cs"),
                  color = "steelblue", linewidth = 1) +
      geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed", linewidth = 1) +
      labs(x = "Fitted values", y = "Observed values") +
      theme_plain()
    ggsave(file.path(diag_folder, "obs_vs_fitted_smooth.png"), p_smooth, width = gg_width, height = gg_height, dpi = 300)
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
notify_macos <- function(title, notif_message, sound = "default") {
  cmd <- sprintf(
    'display notification "%s" with title "%s" sound name "%s"',
    notif_message, title, sound
  )
  system(sprintf("osascript -e '%s'", cmd))
}

notify_macos(
  title = "Model diagnostics finished",
  notif_message = sprintf("%d models done, %d failed. Output in %s", n_ok, n_failed, output_folder)
)
