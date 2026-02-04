# STAC endpoint - Updates may influence code operation
stac_location = "https://data.geo.admin.ch/api/stac/v0.9/"

# Directory holding intermediate annual min-max datasets - this folder should be empty
existing_data = None

# Forest mask location - this should point to the downloaded forest mask.
forest_mask = r"DiscoCH/data/CH_NoValue_255.tif"

# Template raster with NaN extent (same resolution and CRS) - this should point to the downloaded template.
ch_template = r"DiscoCH/data/CH_NoValue_255.tif"

# Trained discoloration model - this should point to the downloaded model pickle file.
disco_model = r"DiscoCH/data/empirical_discoloration_model_pipeline_2025_6_2.pkl"

# Output directory - an empty folder where you want to save model outputs (.tif and .png files for each valid date)
output_dir = None

# Optional saving of normalized VIs - Only change this to true if you want normalized VIs rather than disco probability
save_norm_vi = False

# Bounding box for a region of interest (None for the full country) - Do not change for the Example
# bounding_box = (2691388, 1285780, 2697392, 1290313)  # Schaffhausen Region 1
# bounding_box = (2688769, 1286490, 2691176, 1288144)  # Schaffhausen Region 2
