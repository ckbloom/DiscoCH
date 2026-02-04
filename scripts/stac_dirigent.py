from src.disco_ch.stac_pull import new_image_check, load_minmax_rasters, update_vi_min_max, build_template
import rioxarray as rxr

"""
Welcome to the STAC Discoloration Model Application tool
This code runs a pipeline that:
1. Pulls lists of images from the swisstopo SwissEO STAC and produces 10m annual Min-Max rasters for five VIs: 
CCI, CIre, NDVI, NDMI, and EVI
2. Annually normalizes rasters at a desired timestep/s
3. Applies a trained discoloration model to the normalized rasters

Define dates and file locations below and run the code to predict discoloration

"""

# Define your dates of interest (YYYY-MM-DD)
start_date = '2018-03-01'
end_date = '2018-08-18'

# Local directory with existing data
existing_data = r"..\DiscoCH\output\minmax_demo"

# If needed, modify the STAC location
stac_location = 'https://data.geo.admin.ch/api/stac/v0.9/'

# Forest Mask
forest_mask = r"..\DiscoCH\data\SEO_DRAINS_Forest_10m_Mask.tif"

# NaN value template for initializing min and max values
ch_template = r"..\DiscoCH\data\NoValue_CH_Extent.tif"

output = r"..\DiscoCH\output\demo"

disco_model = r'..\DiscoCH\data\empirical_discoloration_model_pipeline_2025_6_2.pkl'

bounding_box = (2688769, 1286490, 2691176, 1288144)
# bounding_box = (2691388, 1285780, 2697392, 1290313)  # Alternate Schaffhausen Region

save_norm_vi = False

run_incremental_normalization = True


# Run the code
if __name__ == '__main__':

    # Initialize variables
    vi_min, vi_max = None, None

    # Get year from date of interest
    year_of_interest = int(start_date.split("-")[0])

    # Check for existing min max metadata and new datasets from STAC
    print('Checking for Min Max data')
    items_to_process = new_image_check(start_date, end_date, existing_data, stac_location)

    # Create a no data template for the region of interest
    if bounding_box is not None:
        template = build_template(ch_template, bounding_box, True)
    else:
        template = rxr.open_rasterio(ch_template).squeeze(drop=True).astype("float32")
        template = template.where(template != 255)  # Converts 255 to NaN

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