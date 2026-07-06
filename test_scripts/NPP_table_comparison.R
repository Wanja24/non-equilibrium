library(terra)
library(sf)
# extract lon/lat from .geo JSON string
library(jsonlite)

setwd("~/Library/CloudStorage/OneDrive-LeuphanaUniversity/Research/Non-equilibrium/scripts")

# Load GeoTIFFs
npp1 <- rast("NPP_test_17_06_modis.tif")
npp2 <- rast("NPP_test_17_06_wgs84.tif")

# Quick inspection
npp1
npp2

par(mfrow = c(2, 1))  # 1 row, 2 columns

plot(npp1,
     main = "NPP Test 1",
     col = viridis::viridis(100))

plot(npp2,
     main = "NPP Test 2",
     col = viridis::viridis(100))


# Read exported table
df <- read.csv("NPP_test_17_06_modis_wgs84latlon.csv")

View(df)

dx <- median(diff(sort(unique(df$x))))
dy <- median(diff(sort(unique(df$y))))

ext <- ext(
  min(df$x) - dx/2,
  max(df$x) + dx/2,
  min(df$y) - dy/2,
  max(df$y) + dy/2
)

r_template <- rast(ext, resolution = c(dx, dy), crs = "EPSG:4326")

v <- vect(df, geom = c("longitude", "latitude"), crs = "EPSG:4326")

r_ll <- rasterize(
  v,  
  r_template,
  field = "Npp",
  fun = "mean",   # important if duplicates exist
  background = NA
)

plot(r_ll)


# MODIS
geo <- fromJSON(df$.geo)

df$lon <- geo$coordinates[,1]
df$lat <- geo$coordinates[,2]

# convert to spatial points
pts_ll <- vect(df, geom = c("lon", "lat"), crs = "EPSG:4326")

# estimate resolution
dx <- min(diff(sort(unique(df$lon))))
dy <- min(diff(sort(unique(df$lat))))

r_ll <- rast(
  xmin = min(df$lon) - dx/2,
  xmax = max(df$lon) + dx/2,
  ymin = min(df$lat) - dy/2,
  ymax = max(df$lat) + dy/2,
  resolution = c(dx, dy),
  crs = "EPSG:4326"
)

r_ll <- rasterize(pts_ll, r_ll, field = "Npp")

plot(r_ll)


# Convert modis to wgs84 and extract table
npp1_wgs84 <- project(
  npp1,
  "EPSG:4326",
  method = "bilinear"
)
plot(npp1_wgs84)

npp_tbl <- as.data.frame(
  npp1_wgs84,
  xy = TRUE,
  cells = TRUE,
  na.rm = NA # cells that have NA values in all layers are removed
)

head(npp_tbl)
ncell(npp1)
ncell(npp1_wgs84)
global(!is.na(npp1_wgs84), "sum")
global(!is.na(npp1), "sum")
colSums(!is.na(npp_tbl))
colSums(!is.na(npp_tbl_modis))
colSums(!is.na(df))
plot(npp1[[2]])       # original Npp
plot(npp1_wgs84[[2]]) # reprojected Npp
plot(!is.na(npp1$Gpp) & is.na(npp1$Npp)) # we have more gpp than npp values bc it somehow has values in the oceans too; i.e. ALL FINE


dx <- median(diff(sort(unique(npp_tbl$x))))
dy <- median(diff(sort(unique(npp_tbl$y))))

ext <- ext(
  min(npp_tbl$x) - dx/2,
  max(npp_tbl$x) + dx/2,
  min(npp_tbl$y) - dy/2,
  max(npp_tbl$y) + dy/2
)

r_template <- rast(ext, resolution = c(dx, dy), crs = "EPSG:4326")

v <- vect(npp_tbl, geom = c("x", "y"), crs = "EPSG:4326")

r_ll <- rasterize(
  v,  
  r_template,
  field = "Npp",
  fun = "mean",   # important if duplicates exist
  background = NA
)

plot(r_ll)

'write.csv(
  npp_tbl,
  "NPP_test_17_06_downloaded_modis_pixels.csv",
  row.names = FALSE
)'

dim(npp_tbl)
dim(df)

npp_tbl_modis <- as.data.frame(
  npp1,
  xy = TRUE,
  cells = TRUE,
  na.rm = NA # cells that have NA values in all layers are removed
)

head(npp_tbl_modis)
