# Date of interest
date_of_interest = "2018-08-29"
year_of_interest = int(date_of_interest.split("-")[0])

# STAC endpoint
stac_location = "https://data.geo.admin.ch/api/stac/v0.9/"

# Directory holding annual min-max products
existing_data = r"B:/minmax_demo"

# Small-area forest mask (subset raster)
forest_mask = r"G:\1_cbloom\Projects\2025_01_12_VHI\DiscoCH\data/SEO_DRAINS_Forest_10m_Mask.tif"

# Template raster with NaN extent (same resolution and CRS)
ch_template = r"G:\1_cbloom\Projects\2025_01_12_VHI\DiscoCH\data\NoValue_CH_Extent.tif"

# Output directory
output_dir = r"G:\1_cbloom\Projects\2025_01_12_VHI\DiscoCH\output/demo"

# Trained discoloration model
disco_model = r"G:\1_cbloom\Projects\2025_01_12_VHI\DiscoCH\data/sen_disco_model/2025_03_12/SVC/empirical_model_pipeline_2025_6_2_1dfdb58bc3_dirty.pkl"

# Optional saving of normalized VIs
save_norm_vi = False

# Bounding box for a region of interest (None for the full country)
# bounding_box = (2688769, 1286490, 2691176, 1288144)  # SH1
bounding_box = (2691388, 1285780, 2697392, 1290313)  # SH2

# Runs normalizations and model application after every new min max value is added
run_incremental_normalization = True
