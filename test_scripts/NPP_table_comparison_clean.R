############################################################
# SETUP
############################################################

library(terra)
library(sf)
library(jsonlite)
library(viridis)
#library(dplyr)

setwd("/Volumes/TOSHIBA EXT/non-equilibrium/data")
setwd("/Users/Wanja/Documents/non-equilibrium_data")

############################################################
# LOAD RASTERS (MODIS + WGS84 REPROJECTED)
# & EXPORTED TABLE (EE SAMPLE OUTPUT)
############################################################

npp_modis_tif <- rast("npp/NPP_2011-01-01.tif") # NPP_test_17_06_modis.tif, NPP_5km_2001_1_test_24_06_modis.tif, NPP_5km_2001_1_wholeworld_25_06.tif
npp_wgs84_tif <- rast("test/NPP_test_17_06_wgs84.tif")
df_ee <- read.csv("test/NPP_5km_2001_1_smallregion_25_06.csv") # "NPP_test_17_06_modis_wgs84latlon.csv"
world_borders <- vect("world-administrative-boundaries/world-administrative-boundaries.shp")
region_ee <- read.csv("trash/Region_test.csv")
rangeland_tif <- rast("rangeland_mask_new/Rangeland_mask_2024-01-01.tif")
climate_tif <- rast("test/Climate_yearly_2001-01-01.tif")
climate_native <- rast("/Users/Wanja/Downloads/Climate_northwest_2001_01_native_crs.tif")
climate_modis <- rast("climate_monthly/Climate_monthly_2001-01-01.tif") # "/Users/Wanja/Downloads/Climate_northwest_masked-9999_monthly_2001-01-01.tif"
elevation_raw <- rast("/Users/Wanja/Downloads/Elevation_raw_2010-01-01.tif")
elevation_resampled <- rast("/Users/Wanja/Downloads/Elevation_resampled_masked_2010-01-01.tif")
veg_period <- rast("/Users/Wanja/Documents/repos/non-equilibrium/test_scripts/vegetation_period_2024.tif")
climate_yearly <- rast('climate_yearly/Climate_yearly_2002.tif')
climate_veg_period <- rast('climate_yearly/Climate_vegetation_period_2002.tif')

# Quick inspection
npp_modis_tif
npp_wgs84_tif
climate_tif
View(df_ee)
world_borders
plot(world_borders)

############################################################
# SIDE-BY-SIDE PLOT OF RASTERS
############################################################

par(mfrow = c(2, 1))

plot(
  npp_modis_tif,
  main = "NPP MODIS (original)",
  col = viridis(100)
)

plot(
  npp_wgs84_tif,
  main = "NPP WGS84 GeoTIFF",
  col = viridis(100)
)

par(mfrow = c(1, 1))

############################################################
# PROJECT MODIS TO WGS84 (RASTER SIDE)
############################################################

npp_modis_wgs84 <- project(
  npp_modis_tif,
  "EPSG:4326",
  method = "bilinear"
)

plot(npp_modis_wgs84, main = "MODIS projected to WGS84")

############################################################
# EXTRACT TABLE FROM REPROJECTED RASTER
############################################################

df_wgs84 <- as.data.frame(
  npp_modis_wgs84,
  xy = TRUE,
  cells = TRUE,
  na.rm = NA
)

head(df_wgs84)

############################################################
# BASIC COMPARISONS (RASTER + TABLE COVERAGE)
############################################################

ncell(npp_modis_tif)
ncell(npp_modis_wgs84)
ncell(climate_modis_wgs84)

global(!is.na(npp_modis_tif), "sum")
global(!is.na(npp_modis_wgs84), "sum")

colSums(!is.na(df_ee))
colSums(!is.na(df_wgs84))

dim(df_ee)
dim(df_wgs84)
dim(df_climate_wgs84)

plot(!is.na(npp_modis_tif$Gpp) & is.na(npp_modis_tif$Npp)) # we have more gpp than npp values bc it somehow has values in the oceans too; i.e. ALL FINE


############################################################
# RECONSTRUCTION FROM WGS84 TABLE
############################################################
dx_tbl <- median(diff(sort(unique(df_wgs84$x))))
dy_tbl <- median(diff(sort(unique(df_wgs84$y))))

ext_tbl <- ext(
  min(df_wgs84$x) - dx_tbl / 2,
  max(df_wgs84$x) + dx_tbl / 2,
  min(df_wgs84$y) - dy_tbl / 2,
  max(df_wgs84$y) + dy_tbl / 2
)

r_template_tbl <- rast(
  ext_tbl,
  resolution = c(dx_tbl, dy_tbl),
  crs = "EPSG:4326"
)

# or use tif as template
r_template <- npp_modis_wgs84 #npp_modis_wgs84

v_tbl <- vect(df_wgs84, geom = c("x", "y"), crs = "EPSG:4326")

r_recon_tbl <- terra::rasterize(
  v_tbl,
  r_template_tbl,
  field = "Npp",
  fun = "mean",
  background = NA
)

par(mfrow = c(2, 1))

plot(npp_modis_wgs84$Npp, main = "MODIS projected to WGS84")

plot(r_recon_tbl, main = "Reconstructed raster (from extracted table)")

par(mfrow = c(1, 1))

############################################################
# RECONSTRUCTION FROM EXPORTED EE TABLE
############################################################

v_df <- vect(df_ee, geom = c("longitude", "latitude"), crs = "EPSG:4326")

r_recon_df <- terra::rasterize(
  v_df,
  r_template_tbl,
  field = "Npp",
  fun = "mean",
  background = NA
)

plot(r_recon_df, main = "Fixed reconstruction (aligned grid)")

############################################################
# SIDE-BY-SIDE COMPARISON OF ALL RECONSTRUCTIONS
############################################################

par(mfrow = c(2, 1))

plot(r_recon_df, main = "From exported ee table (sampled from modis grid)", col = viridis(100))
plot(r_recon_tbl, main = "From wgs84 table (sampled from true wgs84) ", col = viridis(100))

par(mfrow = c(1, 1))

############################################################
# EXPORT TABLE (OPTIONAL)
############################################################

'write.csv(
  df_wgs84,
  "NPP_test_17_06_downloaded_modis_pixels.csv",
  row.names = FALSE
)'

############################################################
# Merge country/region to table
############################################################

continent_raster <- rasterize(
  world_borders,
  npp_modis_wgs84,   # or r_recon_tbl
  field = "continent"   # country name field (adjust if needed)
)

region_raster <- rasterize(
  world_borders,
  npp_modis_wgs84,   # or r_recon_tbl
  field = "region"   # country name field (adjust if needed)
)

npp_stack <- c(npp_modis_wgs84, continent_raster, region_raster)
plot(npp_stack)

names(npp_stack) <- c("GPP", "NPP", "NPP_QC", "continent", "region")

npp_raster_table <- as.data.frame(
  npp_stack,
  xy = TRUE,
  na.rm = NA
)

head(npp_raster_table)
dim(npp_raster_table)
colSums(!is.na(npp_raster_table))
View(npp_raster_table)

############################################################
# Merge rangeland (exported as tif from ee) to table
############################################################

rangeland_wgs84 <- project(
  rangeland_tif,
  "EPSG:4326",
  method = "near"
)

plot(rangeland_wgs84, main = "Region projected to WGS84")

npp_stack <- c(npp_modis_wgs84, continent_raster, region_raster, rangeland_wgs84)
plot(npp_stack)

names(npp_stack) <- c("GPP", "NPP", "NPP_QC", "continent", "region", "rangeland")

npp_raster_table <- as.data.frame(
  npp_stack,
  xy = TRUE,
  na.rm = NA
)

head(npp_raster_table)
dim(npp_raster_table)
colSums(!is.na(npp_raster_table))
sum(npp_raster_table$rangeland)
View(npp_raster_table)


############################################################
# Merge region (ee exported) to table
############################################################

# NPP table
sum(duplicated(df_ee[, c("longitude", "latitude")]))

# Region table
sum(duplicated(region_ee[, c("longitude", "latitude")]))

n_overlap <- inner_join(
  df_ee[, c("longitude", "latitude")],
  region_ee[, c("longitude", "latitude")],
  by = c("longitude", "latitude")
) %>%
  nrow()

n_overlap
nrow(df_ee)
nrow(region_ee)

df_merged <- left_join(
  df_ee,
  region_ee,
  by = c("longitude", "latitude")
)

colSums(is.na(df_merged))

head(df_ee[, c("longitude", "latitude")])
head(region_ee[, c("longitude", "latitude")])

all.equal(
  sort(unique(df_ee$longitude)),
  sort(unique(region_ee$longitude))
)
all.equal(
  sort(unique(df_ee$latitude)),
  sort(unique(region_ee$latitude))
)




# ---- Climate test ----

############################################################
# PROJECT MODIS TO WGS84 (RASTER SIDE)
############################################################

climate_modis_wgs84 <- project(
  climate_tif,
  "EPSG:4326",
  method = "bilinear"
)

plot(climate_modis_wgs84, main = "Climate MODIS projected to WGS84")

############################################################
# EXTRACT TABLE FROM REPROJECTED RASTER
############################################################

df_climate_wgs84 <- as.data.frame(
  climate_modis_wgs84,
  xy = TRUE,
  cells = TRUE,
  na.rm = NA
)

head(df_climate_wgs84)

############################################################
# COMPARE CLIMATE
############################################################
climate_native
climate_modis
crs(climate_native)
crs(climate_modis)

NAflag(climate_modis)
climate_modis[climate_modis == -9999] <- NaN
global(is.na(climate_modis), "sum")

climate_native_wgs84 <- project(
  climate_native,
  "EPSG:4326",
  method = "bilinear"
)

climate_modis_wgs84 <- project(
  climate_modis,
  "EPSG:4326",
  method = "bilinear"
)

plot(climate_native_wgs84)
plot(climate_modis_wgs84)

par(mfrow = c(2, 1))

plot(
  climate_native_wgs84[[5]],
  main = "Climate (from native proj)",
  col = viridis(100),
  range = c(0, 200),
  #breaks = c(50, 100, 150, 200, 2000)
)

plot(
  climate_modis_wgs84[[1]],
  main = "Climate (from modis proj)",
  col = viridis(100),
  range = c(0, 200),
  xlim = c(-180, 0),
  #breaks = c(50, 100, 150, 200, 2000)
)

par(mfrow = c(1, 1))


############################################################
# STACK NPP & CLIMATE
############################################################

npp_modis_wgs84_cropped <- crop(npp_modis_wgs84, ext(climate_modis_wgs84))
climate_modis_wgs84_resampled <- resample(climate_modis_wgs84, npp_modis_wgs84)

ext(climate_modis_wgs84)
ext(npp_modis_wgs84)
ext(npp_modis_wgs84_cropped)
ext(climate_modis_wgs84_resampled)

npp_climate_stack <- c(npp_modis_wgs84, climate_modis_wgs84_resampled)
plot(npp_climate_stack)

npp_climate_table <- as.data.frame(
  npp_climate_stack,
  xy = TRUE,
  na.rm = NA
)

head(npp_climate_table)
dim(npp_climate_table)
colSums(!is.na(npp_climate_table))
sum(complete.cases(npp_climate_table))
View(npp_climate_table)



############################################################
# ELEVATION
############################################################
elevation_raw
plot(elevation_raw)

elevation_resampled
NAflag(elevation_resampled)
elevation_resampled[elevation_resampled == -9999] <- NaN
global(is.na(elevation_resampled), "sum")
plot(elevation_resampled)
plot(elevation_resampled[[1]])

npp_elevation_stack <- c(npp_modis_tif, elevation_resampled)
plot(npp_elevation_stack)

rangeland_tif
plot(rangeland_tif)
