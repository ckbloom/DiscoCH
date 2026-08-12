from src.disco_ch.force_pull import run_tsa_workflow_tiled

"""
Welcome to the TSA Discoloration Model Application tool
This code runs a pipeline that:
1. Scans a directory of 30km grid tile folders (e.g. X0052_Y0064), each
   holding 5 pre-computed multiband VI TIFs (CCI, CIRE, NDVI, NDMI, EVI)
   with one band per acquisition date, and produces 10m annual Min-Max
   rasters per tile for each VI
2. Annually normalizes rasters at a desired timestep/s, per tile
3. Applies a trained discoloration model to the normalized rasters, per tile
4. Mosaics each date's per-tile model output into a single national/AOI raster
Define dates and file locations below and run the code to predict discoloration
"""

# Define your dates of interest (YYYY-MM-DD)
start_date = '2026-03-01'
end_date = '2026-09-30'

# Root directory containing the X####_Y#### tile folders
tsa_root = r"B:\bloomc\DiscoCH_2026_08_03\FORCE\interpolated_vi_rbf_10"
# r"\\speedy16-36\Data_23\FORCE\FORCE_Kingslide\level2\tsa\real_values_flagged"
# r"B:\bloomc\DiscoCH_2026_08_03\FORCE\level3_tss" # r"\\speedy16-36\Data_23\FORCE\FORCE_Kingslide\level3_tss"

vi_pattern_template="{year}*_{vi}_TSS_rbf.tif"

# Local directory for per-tile min/max caches + metadata (a subfolder is
# created per tile automatically, e.g. minmax_tsa/X0052_Y0064)
existing_data_root = r"B:\bloomc\DiscoCH_2026_08_03\FORCE\minmax"

# Forest Mask (national/AOI extent -- each tile clips its own region out
# of this automatically)
forest_mask = r"G:\1_cbloom\Projects\2025_01_12_VHI\DiscoCH\data\DRAINS_Forest_Mask.tif"

# NaN value template for initializing min and max values (national/AOI
# extent, same as forest_mask)
ch_template = r"G:\1_cbloom\Projects\2025_01_12_VHI\DiscoCH\data\CH_NoValue_255.tif"

# Root directory for per-tile outputs + the final mosaic (output_root/mosaic)
output_root = r"B:\bloomc\DiscoCH_2026_08_03\FORCE\output"

disco_model = r'G:\1_cbloom\Projects\2025_01_12_VHI\DiscoCH\data\empirical_discoloration_model_pipeline_2025_6_2.pkl'

# Only dates falling in these months get normalized + run through the
# model (min/max still updates for every processed date)
model_months = (5, 6, 7)

# Plot each individual tile's per-date output as it's produced
plot_tile_results = False

# Plot the final mosaicked per-date output after all tiles are merged
plot_mosaic_results = False

# Set the VI keys in the dataset
vi_keys_2026 = {
    "NDVI": "NDVI",
    "EVI":  "EVI",
    "NDMI": "NDMI",
    "CIRE": "CIre",
    "CCI":  "CCI",
}
vi_keys_2018 = {
    "NDVI": "NDV",
    "EVI":  "EVI",
    "NDMI": "NDM",
    "CIRE": "CRE",
    "CCI":  "CCI",
}

# Run the code
if __name__ == '__main__':
    # Get year from date of interest
    year_of_interest = int(start_date.split("-")[0])

    print('Running TSA tiled discoloration workflow')
    run_tsa_workflow_tiled(
        root_dir=tsa_root,
        year_of_interest=year_of_interest,
        forest_mask_path=forest_mask,
        template_path=ch_template,
        existing_data_root=existing_data_root,
        output_root=output_root,
        disco_model=disco_model,
        model_months=model_months,
        vi_keys=vi_keys_2026,
        start_date=start_date,
        end_date=end_date,
        plot_tile_results=plot_tile_results,
        plot_mosaic_results=plot_mosaic_results,
        vi_pattern_template=vi_pattern_template,
        max_workers=10,
    )