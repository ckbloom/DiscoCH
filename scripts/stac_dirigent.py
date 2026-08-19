from src.disco_ch.stac_pull import (
    new_image_check, load_minmax_rasters, update_vi_min_max, update_vi_min_max_interpolated,
    build_template,
)
import rioxarray as rxr

"""
Welcome to the STAC Discoloration Model Application tool
This code runs a pipeline that:
1. Pulls lists of images from the swisstopo SwissEO STAC and produces 10m annual Min-Max rasters for five VIs:
CCI, CIre, NDVI, NDMI, and EVI
2. Annually normalizes rasters at a desired timestep/s
3. Applies a trained discoloration model to the normalized rasters

Define dates and file locations below and run the code to predict discoloration

Two modes, toggled by `interpolate` below:
  - interpolate=True (default): mirrors the FORCE+RBF pipeline
    (force_dirigent.py) -- every pulled scene's VI is archived to
    `archive_root` regardless of month (cheap, just builds up history),
    and min/max + the model are only ever run on rbf_interp's delayed
    ensemble-kernel interpolation at a fixed step_days cadence within
    [season_start, season_end]. Pulling from `start_date` well before
    `season_start` costs archiving only -- it just gives the first
    in-season output dates real backward context once interpolation
    starts.
  - interpolate=False: original behavior, unchanged -- every scene folds
    straight into min/max (and, in model_months, the model) the moment
    it's pulled, with no interpolation step.
"""

# Set to False to fall back to the original raw (non-interpolated) pipeline.
interpolate = True

# Define your dates of interest (YYYY-MM-DD). With interpolate=True this can
# safely start well before season_start -- see note above.
start_date = '2026-03-01'
end_date = '2026-08-12'

# Local directory with existing data (min/max cache + metadata)
existing_data = r"B:\bloomc\DiscoCH_2026_08_03\swissEO\test\minmax"

# interpolate=True only: parent folder for the per-VI raw-scene archive
# that rbf_interp reads from (archive_root/<VI>/<date>.tif). Grows across
# runs -- keep it separate from `existing_data` and never delete it
# mid-season, or backward context for RBF is lost.
archive_root = r"B:\bloomc\DiscoCH_2026_08_03\swissEO\test\rbf_archive"

# interpolate=True only: fixed output-date calendar that min/max and the
# model are evaluated on -- see rbf_interp.growing_season_dates(). Keep
# season_start close to model_months' start; the wider [start_date,
# end_date] pull above is what builds RBF's backward context.
season_start = "05-01"
season_end = "08-12"
step_days = 5

# interpolate=True only: rbf_interp.delayed_smooth_one_date() params.
# Leave as None to use rbf_interp's own defaults.
rbf_widths = None
rbf_wait_days = None
rbf_max_backward_gap_days = None
rbf_max_radius_days = None

# interpolate=True only: triplet-residual despiking (see
# rbf_interp.despike_daily_cubes_union()). A date is removed for a pixel
# if ANY of the 5 VIs' own residual test flags it -- contamination
# (residual cloud/shadow/atmospheric distortion) corrupts the shared raw
# bands, so one VI's detection protects all of them. Set despike=False to
# fall back to the raw (unfiltered) archive.
despike = True
despike_threshold_factor = None  # None -> rbf_interp's default (3.0)
despike_max_iter = None  # None -> rbf_interp's default (20)

# If needed, modify the STAC location
stac_location = 'https://data.geo.admin.ch/api/stac/v0.9/'

# Forest Mask
forest_mask = r"G:\1_cbloom\Projects\2025_01_12_VHI\DiscoCH\data\DRAINS_Forest_Mask.tif"

# NaN value template for initializing min and max values
ch_template = r"G:\1_cbloom\Projects\2025_01_12_VHI\DiscoCH\data\CH_NoValue_255.tif"

output = r"B:\bloomc\DiscoCH_2026_08_03\swissEO\test\output"

disco_model = r'G:\1_cbloom\Projects\2025_01_12_VHI\DiscoCH\data\empirical_discoloration_model_pipeline_2025_6_2.pkl'

bounding_box = (2688769, 1286490, 2691176, 1288144)  # None

save_norm_vi = False

run_incremental_normalization = True

# Only dates falling in these months get normalized + run through the
# model (min/max still updates for every processed date/output date)
model_months = (5, 6, 7)


# Run the code
if __name__ == '__main__':

    # Initialize variables
    vi_min, vi_max = None, None

    # Get year from date of interest
    year_of_interest = int(start_date.split("-")[0])

    # Create a no data template for the region of interest
    if bounding_box is not None:
        template = build_template(ch_template, bounding_box, True)
    else:
        template = rxr.open_rasterio(ch_template).squeeze(drop=True).astype("float32")
        template = template.where(template != 255)  # Converts 255 to NaN

    if interpolate:
        # New scenes are tracked against archive_root (not existing_data)
        # here -- "already processed" means "already archived", regardless
        # of whether an output date has been through min/max + the model yet.
        print('Checking for new STAC scenes to archive')
        items_to_process = new_image_check(start_date, end_date, archive_root, stac_location)

        vi_min, vi_max = update_vi_min_max_interpolated(
            items_to_process, year_of_interest, existing_data, forest_mask, template, bounding_box,
            archive_root,
            disco_model=disco_model,
            output_dir=output,
            model_months=model_months,
            plot_results=False,
            widths=rbf_widths,
            wait_days=rbf_wait_days,
            max_backward_gap_days=rbf_max_backward_gap_days,
            max_radius_days=rbf_max_radius_days,
            season_start=season_start,
            season_end=season_end,
            step_days=step_days,
            despike=despike,
            despike_threshold_factor=despike_threshold_factor,
            despike_max_iter=despike_max_iter,
        )
    else:
        # Check for existing min max metadata and new datasets from STAC
        print('Checking for Min Max data')
        items_to_process = new_image_check(start_date, end_date, existing_data, stac_location)

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
                              output_dir=output,
                              model_months=model_months,
                              plot_results=False)