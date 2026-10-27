# Model validation script - overnight - version with separate R subprocess

# Section 7 & 8: Model interpretation & validation ----

# Load libraries
library(callr) # to start R subprocess

# Set the working directory and import the data.
setwd("/Users/Wanja/Documents/non-equilibrium_data")

# Folder with saved models
output_folder <- "output/output_overnight"

# Models 
model_files <- list.files(output_folder, pattern = "^mod[0-9a-zA-Z_]+\\.rds$", full.names = FALSE)
model_files <- c("mod4_wiggly_20261016.rds", "mod5_withoutSpace_20261016.rds")
model_names <- gsub("\\.rds$", "", model_files)

# R file with diagnostic functions
helpers_file <- "/Users/Wanja/Documents/repos/non-equilibrium/test_scripts/model_diagnostic_functions.R"

# Run across all saved models
n_ok <- 0
n_failed <- 0

for (mod_name in model_names) {
  cat("\n--- Running diagnostics for", mod_name, "---\n")
  
  ok <- tryCatch(
    callr::r(
      function(mod_name, output_folder, helpers_file) {
        source(helpers_file)
        mod <- readRDS(file.path(output_folder, paste0(mod_name, ".rds")))
        res <- run_model_diagnostics(mod, mod_name, output_folder)
        !is.null(res)
      },
      args = list(mod_name = mod_name,
                  output_folder = output_folder,
                  helpers_file = helpers_file),
      show = TRUE
    ),
    error = function(e) {
      message("Failed for ", mod_name, ": ", conditionMessage(e))
      FALSE
    }
  )
  
  if (isTRUE(ok)) n_ok <- n_ok + 1 else n_failed <- n_failed + 1
}

cat("\nDone:", n_ok, "ok,", n_failed, "failed\n")


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
