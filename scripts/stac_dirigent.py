from src.disco_ch.stac_pull import (new_image_check, load_minmax_rasters, update_vi_min_max, pull_closest_from_stac,
                                    normalize_vis, build_template)

from src.disco_ch.apply_model_stac import apply_disco
import os

"""
Welcome to the STAC Discoloration Model Application tool
This code runs a pipeline that:
1. Pulls lists of images from the swisstopo SwissEO STAC and produces 10m annual Min-Max rasters for five VIs: 
CCI, CIre, NDVI, NDMI, and EVI
2. Annually normalizes rasters at a desired timestep
3. Applies a trained discoloration model to the normalized rasters

Define a date and file locations below and run the code to predict discoloration (at the closest available date)

"""

# Define your date of interest
date_of_interest = '2018-08-18'

# Local directory with existing data
existing_data = r"B:\bloomc\minmax_demo"

# If needed, modify the STAC location
stac_location = 'https://data.geo.admin.ch/api/stac/v0.9/'

# Forest Mask
forest_mask = r"G:\1_cbloom\Projects\2025_01_12_VHI\DiscoCH\data\SEO_DRAINS_Forest_10m_Mask.tif"

# NaN value template for initializing min and max values
ch_template = r"G:\1_cbloom\Projects\2025_01_12_VHI\DiscoCH\data\NoValue_CH_Extent.tif"

output = r"G:\1_cbloom\Projects\2025_01_12_VHI\DiscoCH\output\demo"

disco_model = (r'G:\1_cbloom\Projects\2025_01_12_VHI\DiscoCH\data\sen_disco_model\2025_03_12\SVC'
               r'\empirical_model_pipeline_2025_6_2_1dfdb58bc3_dirty.pkl')
# disco_model=None

bounding_box = (2688769, 1286490, 2691176, 1288144)  # SH

save_norm_vi = False

run_incremental_normalization = True


# Run the code
if __name__ == '__main__':

    # Step 1: Check the SwissEO STAC for new rasters to process for Min-Max and process them if needed
    ###################################################################################################################
    # Initialize variables
    vi_min, vi_max = None, None

    # Get year from date of interest
    year_of_interest = int(date_of_interest.split("-")[0])

    # Check for existing min max metadata and new datasets from STAC
    print('Checking for Min Max data')
    items_to_process, existing_meta, processed_dates = new_image_check(date_of_interest, existing_data, stac_location)

    # Create a no data template for the region of interest
    if bounding_box is not None:
        template = build_template(ch_template, bounding_box)
    else:
        template = ch_template

    # If no new Min Max processing is needed, load vi min and max rasters
    if items_to_process is None:
        try:
            vi_min, vi_max = load_minmax_rasters(year_of_interest, existing_data)
        except FileNotFoundError:
            print("Metadata said processed, but min-max rasters are missing — delete the metadata file and restart")

    # Process new dates into annual Min Max VIs
    else:
        update_vi_min_max(items_to_process, year_of_interest, existing_data, forest_mask, template, bounding_box,
                          run_after_each_update=run_incremental_normalization,
                          disco_model=disco_model,
                          output_dir=output)
        vi_min, vi_max = load_minmax_rasters(year_of_interest, existing_data)

    # # Step 2: Normalize the closest raster to the date of interest
    # ###################################################################################################################
    # print('Processing Normalized Vegetation Indices')
    # # Pulls the closest date within the year to the date of interest
    # closest_data = pull_closest_from_stac(date_of_interest, stac_location)
    #
    # # Normalize the data at the closest date
    # normalized_vis = normalize_vis(closest_data, vi_min, vi_max, forest_mask, bounding_box)
    #
    # if save_norm_vi:
    #     for vi_name, da in normalized_vis.items():
    #         output_path = os.path.join(output, f"{vi_name}_{date_of_interest}.tif")
    #         da_transpose = da.transpose('band', 'y', 'x')
    #         da_transpose.rio.to_raster(output_path)
    #         print(f"Saved {vi_name} to {output_path}")
    #
    # # Step 3: Apply Discoloration Model to normalized VIs
    # ##################################################################################################################
    # output_path = f"Disco_Proba_{date_of_interest}.tif"
    #
    # apply_disco(normalized_vis, disco_model, output_path)

    # import os
    # import rioxarray
    # import glob
    #
    #
    # def rebuild_normalized_vis(input_dir, date_of_interest):
    #     """
    #     Loads saved GeoTIFFs back into a dictionary of xarray DataArrays.
    #     """
    #     normalized_vis = {}
    #
    #     # Define the expected names (as saved in your previous step)
    #     # The glob pattern looks for anything ending in {date_of_interest}.tif
    #     search_pattern = os.path.join(input_dir, f"*_{date_of_interest}.tif")
    #     tif_files = glob.glob(search_pattern)
    #
    #     for tif_path in tif_files:
    #         # Extract the vi_name from the filename (e.g., "NDVI" from "NDVI_2023-01-01.tif")
    #         file_name = os.path.basename(tif_path)
    #         vi_name = file_name.replace(f"_{date_of_interest}.tif", "")
    #
    #         # Load the data
    #         # .squeeze() removes the 'band' dimension to return to (y, x) shape
    #         # .drop_vars("band") removes the band coordinate entirely
    #         da = rioxarray.open_rasterio(tif_path).squeeze().drop_vars("band")
    #
    #         normalized_vis[vi_name] = da
    #         print(f"Loaded {vi_name} from {file_name}")
    #
    #     return normalized_vis
    #
    #
    # saved_path = r'G:\1_cbloom\Projects\2024_08_08_Discoloration_Paper\Discoloration_2025_12_03\output\SEO_Norm_VIs'
    # normalized_vis = rebuild_normalized_vis(saved_path, date_of_interest)
    #
    # print('Applying Discoloration Model')

    # output_path = f"Disco_Proba_{date_of_interest}.tif"
    #
    # apply_disco(normalized_vis, disco_model, output_path)
