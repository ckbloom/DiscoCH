# Date of interest - The last date in the year which the model will be applied to
date_of_interest = "2018-08-29"
year_of_interest = int(date_of_interest.split("-")[0])

# STAC endpoint - Updates may influence code operation
stac_location = "https://data.geo.admin.ch/api/stac/v0.9/"

# Directory holding intermediate annual min-max datasets - this folder should be empty
existing_data = r"B:/minmax_demo"

# Forest mask location - this should point to the downloaded forest mask.
forest_mask = r"G:\1_cbloom\Projects\2025_01_12_VHI\DiscoCH\data\DRAINS_Forest_Mask.tif"

# Template raster with NaN extent (same resolution and CRS) - this should point to the downloaded template.
ch_template = r"G:\1_cbloom\Projects\2025_01_12_VHI\DiscoCH\data\CH_NoValue_255.tif"

# Trained discoloration model - this should point to the downloaded model pickle file.
disco_model = r"G:\1_cbloom\Projects\2025_01_12_VHI\DiscoCH\data\empirical_discoloration_model_pipeline_2025_6_2.pkl"

# Output directory - an empty folder where you want to save model outputs (.tif and .png files for each valid date)
output_dir = r"G:\1_cbloom\Projects\2025_01_12_VHI\DiscoCH\output\demo"

# Optional saving of normalized VIs - Only change this to true if you want normalized VIs rather than disco probability
save_norm_vi = False

# Bounding box for a region of interest (None for the full country) - Do not change for the Example
bounding_box = (2691388, 1285780, 2697392, 1290313)  # Schaffhausen Region 1
# bounding_box = (2688769, 1286490, 2691176, 1288144)  # Schaffhausen Region 2

# Runs normalizations and model application after every new min max value is added - Set to True for the Example
run_incremental_normalization = True
