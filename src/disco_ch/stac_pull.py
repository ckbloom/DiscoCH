import os
import gc
import json
import time
import pystac_client
from datetime import datetime, UTC
import xarray as xr
import rioxarray as rxr
import requests
import warnings
import numpy as np
import shutil
import dask
import matplotlib.pyplot as plt
import glob
import math
import contextily as cx
from matplotlib.colors import LinearSegmentedColormap
from rioxarray.exceptions import NoDataInBounds
from disco_ch.apply_model_stac import apply_disco

# Suppress specific warning
warnings.filterwarnings(
    "ignore",
    message="angle from rectified to skew grid parameter lost in conversion to CF",
    category=UserWarning
)
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in divide",
    category=RuntimeWarning
)


# Vegetation Indices - https://force-eo.readthedocs.io/en/latest/components/higher-level/tsa/indices.html#indices
def ndv(nir, red):
    return (nir - red) / (nir + red)  # nir: B08, red: B04


def ndm(nir, swir1):
    return (nir - swir1) / (nir + swir1)  # nir: B08, swir1: B11


def evi(nir, red, blue):
    return 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))  # nir: B08, red: B04, blue: B02


def cire(rededge3, rededge1):
    return (rededge3 / rededge1) - 1  # rededge3: B07 (Not Available through SwissEO!), rededge1: B05


def cci(green, red):
    return (green - red) / (green + red)  # green: B03, red: B04


def print_metadata(assets, metadata_key):
    """
    Prints relevant metadata from a single file
    :param assets:
    :param metadata_key:
    :return:
    """
    metadata_asset = assets[metadata_key]
    response = requests.get(metadata_asset.href)
    metadata = response.json()  # directly parse JSON from response

    print('10m Bands:')
    bands_10m = metadata["BANDS-10M"]["BANDS"]
    for b in bands_10m:
        print(b["id"])

    print('20m Bands:')
    bands_20m = metadata["BANDS-20M"]["BANDS"]
    for b in bands_20m:
        print(b["id"])

    # Masks (10m)
    print('Masks (10m):')
    masks_10m = metadata.get("MASKS-10M", {}).get("BANDS", [])
    for b in masks_10m:
        print(b["id"])

    # Cloud probability (10m)
    print('Cloud probability (10m):')
    clouds_10m = metadata.get("CLOUDPROBABILITY-10M", {}).get("BANDS", [])
    for b in clouds_10m:
        print(b["id"])


def pull_from_stac(stac_loc='https://data.geo.admin.ch/api/stac/v0.9/', year=2018, date=None):
    """
    Connect to the swisstopo stac and collect datasets
    :param date:
    :param stac_loc:
    :param year:
    :return:
    """
    # Create a connection with the STAC data collection
    service = pystac_client.Client.open(stac_loc)
    service.add_conforms_to("COLLECTIONS")
    service.add_conforms_to("ITEM_SEARCH")

    if date is None:
        # Define start and end dates in the year of interest
        start_date = f'{year}-04-01'
        end_date = f'{year}-09-30'
    else:
        # Define start and end dates in the year of interest
        start_date = f'{year}-03-01'
        end_date = date

    # Filter by start and end date
    item_search = service.search(collections=['ch.swisstopo.swisseo_s2-sr_v100'], datetime=f'{start_date}/{end_date}')

    # Create a list of the data available within the time window
    item_list = list(item_search.items())
    print(f"Found {len(item_list)} images in {year}")
    item_list = sorted(item_list, key=lambda x: datetime.strptime(x.id, "%Y-%m-%dt%H%M%S"))

    return item_list


def load_minmax_metadata(year, folder="minmax"):
    """
    Loads metadata created to support min max calculation in application
    :param year:
    :param folder:
    :return:
    """
    path = os.path.join(folder, f"vi_minmax_{year}_meta.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def save_minmax_metadata(year, processed_dates, folder="minmax"):
    """
    Saves information on the dates that have been processed for min-max
    :param year:
    :param processed_dates:
    :param folder:
    :return:
    """
    os.makedirs(folder, exist_ok=True)

    metadata = {
        "year": year,
        "processed_dates": processed_dates,
        "last_updated": datetime.now(UTC).isoformat()
    }

    path = os.path.join(folder, f"vi_minmax_{year}_meta.json")
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)


def save_minmax_rasters(vi_min, vi_max, year, folder="minmax"):
    """
    Saves the min-max rasters
    :param vi_min:
    :param vi_max:
    :param year:
    :param folder:
    :return:
    """
    os.makedirs(folder, exist_ok=True)

    for k in vi_min:
        min_temp = f"{folder}/{k}_min_{year}_temp.tif"
        max_temp = f"{folder}/{k}_max_{year}_temp.tif"

        # Remove existing files first (Windows-safe)
        if os.path.exists(min_temp):
            os.remove(min_temp)
        if os.path.exists(max_temp):
            os.remove(max_temp)

        vi_min[k].rio.to_raster(min_temp)
        vi_max[k].rio.to_raster(max_temp)

        min_final = f"{folder}/{k}_min_{year}.tif"
        max_final = f"{folder}/{k}_max_{year}.tif"

        # Atomically move temp to final location
        shutil.move(min_temp, min_final)
        shutil.move(max_temp, max_final)


def load_minmax_rasters(year, folder="minmax", chunks='default'):
    """
    Loads existing min-max rasters
    :param chunks:
    :param year:
    :param folder:
    :return:
    """
    vi_min = {}
    vi_max = {}

    if chunks == 'default':
        chunks = {'x': 1024, 'y': 1024}

    for vi in ["NDVI", "EVI", "NDMI", "CIRE", "CCI"]:
        min_path = f"{folder}/{vi}_min_{year}.tif"
        max_path = f"{folder}/{vi}_max_{year}.tif"

        if os.path.exists(min_path):
            vi_min[vi] = rxr.open_rasterio(min_path, chunks=chunks)
            vi_max[vi] = rxr.open_rasterio(max_path, chunks=chunks)

    if len(vi_min) == 0:
        return None, None

    return vi_min, vi_max


def get_item_datetime(item):
    return datetime.strptime(item.id, "%Y-%m-%dt%H%M%S")


def new_image_check(date, existing_data_loc, stac_loc='https://data.geo.admin.ch/api/stac/v0.9/'):
    """
    Checks for new data in the year of interest
    :param date:
    :param existing_data_loc:
    :param stac_loc:
    :return:
    """

    # Get year from date of interest
    year = int(date.split("-")[0])

    # Pull the list of available data
    items = pull_from_stac(stac_loc=stac_loc, year=year, date=date)

    # Open processed data
    existing_meta = load_minmax_metadata(year, existing_data_loc)
    processed_dates = set(existing_meta["processed_dates"]) if existing_meta else set()

    # Find NEW items only
    new_items = [item for item in items if get_item_datetime(item).isoformat() not in processed_dates]

    print(f"Already processed: {len(processed_dates)}")
    print(f"New images to process: {len(new_items)}")

    if len(new_items) == 0:
        return None, None, None
    else:
        return new_items, existing_meta, processed_dates


def load_and_process_assets(assets, forest_mask, bbox=None, band_metadata=False, verbose=0):
    """
    Load bands from a STAC item or assets dict, align 20m → 10m, and return selected bands and valid mask.

    :param verbose:
    :param bbox: (minx, miny, maxx, maxy)
    :param forest_mask:
    :param band_metadata:
    :param assets: A dict of assets
    :return: dict of bands, valid_mask (xr.DataArray)
    """
    tick = time.time()
    key_problem = None

    try:
        # Get asset keys
        bands_10m_key = next(k for k in assets.keys() if k.endswith("bands-10m.tif"))
        bands_20m_key = next(k for k in assets.keys() if k.endswith("bands-20m.tif"))
        mask_key = next(k for k in assets.keys() if k.endswith("masks-10m.tif"))
    except Exception as e:
        key_problem = 1
        bands_10m_key = None
        bands_20m_key = None
        mask_key = None
        print(f"  WARNING: Problem loading the assets within this item! Skipping. {e}")

    if key_problem is not None:
        bands, valid_mask = None, None
    else:
        # Optionally print out the various asset bands
        if band_metadata:
            metadata_key = next((k for k in assets.keys() if k.endswith('metadata.json')), None)
            print_metadata(assets, metadata_key)

        # Load rasters lazily as dask chunks
        bands_10m = rxr.open_rasterio(assets[bands_10m_key].href, masked=True, chunks={"x": 1024, "y": 1024})
        bands_20m = rxr.open_rasterio(assets[bands_20m_key].href, masked=True, chunks={"x": 1024, "y": 1024})
        masks = rxr.open_rasterio(assets[mask_key].href, masked=True, chunks={"x": 1024, "y": 1024})

        # Apply spatial subset
        if bbox is not None:
            try:
                bands_10m = bands_10m.rio.clip_box(*bbox)
                bands_20m = bands_20m.rio.clip_box(*bbox)
                masks = masks.rio.clip_box(*bbox)
            except NoDataInBounds:
                print("  No data in bbox, skipping asset")
                return None, None

        if verbose == 1:
            load_time = time.time()
            print(f'  Loaded Data and Masks in {load_time - tick:.2f} seconds')

        # Select bands and convert to float32 to save memory
        red = bands_10m.sel(band=1).astype("float32", copy=False)
        green = bands_10m.sel(band=2).astype("float32", copy=False)
        blue = bands_10m.sel(band=3).astype("float32", copy=False)
        nir = bands_10m.sel(band=4).astype("float32", copy=False)

        if verbose == 1:
            astype10_time = time.time()
            print(f'  Converted 10m to float32 in {astype10_time - load_time:.2f} seconds')

        # Reproject Match 20m bands (theoretically lazy, but it seems to compute partially)
        swir = bands_20m.sel(band=2).rio.reproject_match(red, dtype="float32")
        rededge = bands_20m.sel(band=3).rio.reproject_match(red, dtype="float32")

        if verbose == 1:
            astype20_time = time.time()
            print(f'  Converted 20m to 10m and float32 in {astype20_time - astype10_time:.2f} seconds')

        # Select masks
        terrain_mask = masks.sel(band=1)
        cloud_and_cloud_shadow_mask = masks.sel(band=2)

        # Create a combined terrain and cloud mask
        valid_mask = ((cloud_and_cloud_shadow_mask != 1) & (terrain_mask <= 63) & (red > 0) &
                      (forest_mask == 1))

        if verbose == 1:
            mask_time = time.time()
            print(f'  Formatted cloud and terrain mask in {mask_time - astype20_time:.2f} seconds')

        # Collect bands and valid mask
        bands = {
            "red": red,
            "green": green,
            "blue": blue,
            "nir": nir,
            "swir": swir,
            "rededge": rededge
        }

        if verbose == 1:
            print('  Executing tasks with Dask')

        # Combine all arrays in a dict for parallel compute
        all_arrays = {**bands, "valid_mask": valid_mask}

        # Compute all arrays in parallel using all available cores
        computed_arrays = dask.compute(*all_arrays.values(), scheduler='threads')

        # Map back to dictionary
        bands = dict(zip(bands.keys(), computed_arrays[:-1]))
        valid_mask = computed_arrays[-1]

        if verbose == 1:
            ex_time = time.time()
            print(f'  Executed all tasks in {ex_time - mask_time:.2f} seconds')

        # Drop the stacks
        del bands_10m
        del bands_20m
        del masks
        gc.collect()

    return bands, valid_mask


def build_template(national_template, bbox):
    """
    Build a nodata template raster for a given bounding box
    using the provided national template. Writes a GeoTIFF
    next to the national template and returns the file path.
    """
    # Load the national template
    national_da = (
        rxr.open_rasterio(national_template)
        .squeeze(drop=True)
        .astype("float32")
    )

    # Crop to the bounding box
    bbox_template = national_da.rio.clip_box(*bbox)

    # Build output path in same directory
    template_dir = os.path.dirname(national_template)
    template_name = os.path.splitext(os.path.basename(national_template))[0]
    out_path = os.path.join(
        template_dir,
        f"{template_name}_bbox_template.tif"
    )

    # Write GeoTIFF
    bbox_template.rio.to_raster(out_path)

    return out_path


def plot_disco_result(raster_path, item_date, output_dir, min_pixel_count=100):
    # 1. Load data
    da = xr.open_dataarray(raster_path).squeeze()
    plot_data = da.where(da != -9999)

    # 2. Check density
    if int(plot_data.count()) < min_pixel_count:
        print("This asset does not have enough valid pixels to plot")
        return False

    # 3. Setup Colors
    disco_colors = [
        '#5A6E50',  # dark moss green
        '#7D9B82',  # muted sage
        '#D7BE6E',  # wheat yellow
        '#AA7850',  # earthy tan
        '#A05A3C'  # soft rust
    ]
    custom_cmap = LinearSegmentedColormap.from_list("disco_smooth", disco_colors)

    # 4. Plot
    # size=8 sets the height; aspect="equal" fixes the stretching
    # alpha=0.7 allows the satellite imagery to peek through the probability map
    plot_obj = plot_data.plot(
        cmap=custom_cmap,
        vmin=0,
        vmax=1,
        add_colorbar=False,
        size=8,
        alpha=1
    )

    ax = plot_obj.axes

    # 5. Add Satellite Imagery
    # cx.providers.Esri.WorldImagery is also an option
    try:
        cx.add_basemap(ax, crs=plot_data.rio.crs.to_string(), source=cx.providers.Esri.WorldTopoMap)
    except Exception as e:
        print(f"  Warning: Could not load basemap: {e}")

    # 6. Colorbar and Labels
    cbar = plt.colorbar(plot_obj, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Discoloration Probability', fontweight='bold')

    ax.set_title(f"Discoloration Probability: {item_date}", fontsize=14)
    ax.axis("off")

    # 7. Save
    plt.savefig(os.path.join(output_dir, f"Disco_Proba_{item_date}.png"), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_disco_grid(raster_dir, pattern="Disco_Proba_*.tif", cols=3):
    """
    Creates a grid of all discoloration maps overlaid on satellite imagery.
    Ensures correct aspect ratio (no stretching) and consistent scaling.
    """
    # 1. Gather all files
    files = sorted(glob.glob(os.path.join(raster_dir, pattern)))
    if not files:
        print("No files found to grid!")
        return

    num_files = len(files)
    rows = math.ceil(num_files / cols)

    # 2. Setup Custom Continuous Colormap
    disco_colors = ['#D4E157', '#AED581', '#FFEB3B', '#FF9800', '#F44336', '#D32F2F']
    custom_cmap = LinearSegmentedColormap.from_list("disco_smooth", disco_colors)

    # 3. Create the Figure
    # We use a larger size per subplot to ensure satellite tiles are readable
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 6))

    # Flatten axes for easy iteration
    if num_files > 1:
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]

    print(f"Generating satellite grid for {num_files} images. This may take a moment...")

    for i, file_path in enumerate(files):
        ax = axes_flat[i]

        # Load and mask
        da = xr.open_dataarray(file_path).squeeze()
        plot_data = da.where(da != -9999)

        # Extract date from filename
        file_name = os.path.basename(file_path)
        date_str = file_name.replace("Disco_Proba_", "").replace(".tif", "")

        # 4. Plot the data with transparency (alpha)
        # We use ax.set_aspect('equal') instead of xarray's size/aspect for subplots
        im = plot_data.plot(
            ax=ax,
            cmap=custom_cmap,
            vmin=0,
            vmax=1,
            add_colorbar=False,
            alpha=0.6,  # Allows satellite imagery to show through
            zorder=2  # Puts the data above the basemap
        )

        ax.set_aspect('equal', adjustable='datalim')

        # 5. Add Satellite Basemap
        try:
            # We use the CRS from the rioxarray object
            crs_str = plot_data.rio.crs.to_string()
            cx.add_basemap(ax, crs=crs_str, source=cx.providers.Esri.WorldImagery, zoom='auto', zorder=1)
        except Exception as e:
            print(f"  Warning: Basemap failed for {date_str}: {e}")

        ax.set_title(f"Date: {date_str}", fontsize=12, fontweight='bold')
        ax.axis("off")

    # 6. Hide unused subplots if files < rows * cols
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis("off")

    # 7. Add a single shared colorbar at the bottom
    # Adjust fraction/pad to fit the bottom of the grid
    cbar = fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.02, pad=0.03)
    cbar.set_label('Discoloration Probability (Continuous Scale)', fontweight='bold', fontsize=14)
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    plt.suptitle("Seasonal Forest Health Summary: Satellite Overlay", fontsize=24, fontweight='bold', y=0.98)

    # 8. Save and Show
    grid_out = os.path.join(raster_dir, "Satellite_Summary_Grid.png")
    plt.savefig(grid_out, dpi=200, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f"Finished! Summary saved to: {grid_out}")


def update_vi_min_max(items_to_process, year_of_interest, existing_data, forest_mask, template, bounding,
                      band_metadata=False, run_after_each_update=False, disco_model=None, output_dir=None):
    """
    Updates (or creates) VI min max rasters and metadata
    :param output_dir:
    :param disco_model:
    :param run_after_each_update:
    :param bounding:
    :param template:
    :param forest_mask:
    :param existing_data:
    :param items_to_process: List of items to process in the year of interest
    :param year_of_interest: Year to process
    :param band_metadata: Print out information from the STAC
    :return:
    """

    # Initialize the vi_min and vi_max
    vi_min = {}
    vi_max = {}

    drains_forest_mask = rxr.open_rasterio(forest_mask)

    # Iterate over the items available in the SwissEO STAC
    for i, item in enumerate(items_to_process):
        print(f"\n Processing {i + 1}/{len(items_to_process)} : {item.id}")

        # Load the assets and extract relevant bands and masks
        assets = item.assets
        bands, valid = load_and_process_assets(assets, drains_forest_mask, bounding, band_metadata=band_metadata)

        # Take care of occasional problems when loading assests (e.g., missing masks etc.)
        if bands is None:
            continue
        else:

            tick = time.time()

            valid = valid.astype('uint8').transpose('band', 'y', 'x')  # Clean up the mask

            # Calculate Vegetation Indices from the bands
            vis = {
                "NDVI": ndv(bands["nir"], bands["red"]),
                "EVI": evi(bands["nir"], bands["red"], bands["blue"]),
                "NDMI": ndm(bands["nir"], bands["swir"]),
                "CIRE": cire(bands["nir"], bands["rededge"]),  # Using NIR as a proxy for Red Edge 3
                "CCI": cci(bands["green"], bands["red"])
            }

            vi_time = time.time()
            print(f'  Created VIs in {vi_time - tick:.2f} seconds')

            # Lazy load old VI min and max if available
            try:
                vi_min, vi_max = load_minmax_rasters(year_of_interest, existing_data, chunks='default')
            except FileNotFoundError:
                vi_min, vi_max = None, None

            # Initialize on first image if no previous VI min max was loaded
            if vi_min is None:
                print(f'  No existing VI min or max data was found. Creating new files.')

                # Load the template raster with the full extent to capture all orbits
                template_raster = rxr.open_rasterio(template).squeeze(drop=True).astype("float32")

                # Initialize vi_min and vi_max from template
                vi_min = {k: template_raster.copy(deep=True) for k in vis.keys()}
                vi_max = {k: template_raster.copy(deep=True) for k in vis.keys()}

                comp_time = time.time()
                print(f'  Initialized annual Min Max from Template in {comp_time - vi_time:.2f} seconds')

            # Reproject valid mask to match vi_min
            valid_aligned = valid.rio.reproject_match(vi_min['NDVI'])

            # Compare and update existing min/max one VI at a time
            for k in vis:
                vi = vis[k]  # .expand_dims('band').transpose('band', 'y', 'x')

                # Reproject current VI to match existing min/max
                vi = vi.rio.reproject_match(vi_min[k]).chunk({'x': 1024, 'y': 1024})

                # Apply the valid mask to the current VI
                vi = vi.where(valid_aligned == 1, np.nan)

                # Lazy min/max
                vi_min[k] = xr.ufuncs.fmin(vi_min[k], vi)
                vi_max[k] = xr.ufuncs.fmax(vi_max[k], vi)

                # Remove infinities lazily
                vi_min[k] = vi_min[k].where(np.isfinite(vi_min[k]), np.nan).transpose('band', 'y', 'x')
                vi_max[k] = vi_max[k].where(np.isfinite(vi_max[k]), np.nan).transpose('band', 'y', 'x')

                # Compute and Save to disk
                save_minmax_rasters({k: vi_min[k]}, {k: vi_max[k]}, year_of_interest, existing_data)

            # Free up memory between runs
            del vis
            gc.collect()

            comp_time = time.time()
            print(f'  Compared to existing Min Max in {comp_time - vi_time:.2f} seconds')

            # Revise the metadata with processed items
            existing_meta = load_minmax_metadata(year_of_interest, existing_data)
            processed_dates = set(existing_meta["processed_dates"]) if existing_meta else set()
            updated_dates = sorted(set(processed_dates) | {get_item_datetime(item).isoformat()})
            save_minmax_metadata(year_of_interest, updated_dates, existing_data)

            backup_time = time.time()
            print(f"  Min/Max saved + metadata updated in {backup_time - comp_time:.2f} seconds")

            if run_after_each_update:
                print("  Running normalization and model application")

                # Reload fresh min max rasters
                vi_min_final, vi_max_final = load_minmax_rasters(
                    year_of_interest, existing_data
                )

                # Normalize only the current item
                normalized_vis = normalize_vis(
                    item,
                    vi_min_final,
                    vi_max_final,
                    forest_mask,
                    bounding
                )

                # Apply discoloration model or save normalized vis
                if output_dir is not None:
                    item_date = get_item_datetime(item).date().isoformat()
                    if disco_model is None:
                        for vi_name, da in normalized_vis.items():
                            out_path = os.path.join(
                                output_dir,
                                f"{vi_name}_{item_date}.tif"
                            )
                            da.transpose("band", "y", "x").rio.to_raster(out_path)

                    # Apply discoloration model
                    else:
                        disco_out = os.path.join(
                            output_dir,
                            f"Disco_Proba_{item_date}.tif"
                        )

                        # Capture the returned DataArray for immediate plotting
                        disco_da = apply_disco(normalized_vis, disco_model, disco_out)

                        # Plotting
                        plot_disco_result(disco_out, item_date, output_dir)

            # Free up memory between items
            del vi_min, vi_max

        # Reload the final rasters to export
        vi_min, vi_max = load_minmax_rasters(year_of_interest, existing_data)

    return vi_min, vi_max


def normalize_vis(closest_data, vi_min, vi_max, forest_mask, bounding):
    """
    Create and normalize vegetation indices by min and max
    :param forest_mask:
    :param closest_data:
    :param vi_min:
    :param vi_max:
    :return:
    """

    drains_forest_mask = rxr.open_rasterio(forest_mask)

    assets = closest_data.assets

    bands, valid = load_and_process_assets(assets, drains_forest_mask, bounding)

    # Take care of occasional problems when loading assests (e.g., missing masks etc.)
    if bands is None:
        print('Data is Unavailable at this Time Step')
    else:
        valid = valid.astype('uint8').transpose('band', 'y', 'x')  # Clean up the mask
        # Reproject valid mask to match vi_min
        valid_aligned = valid.rio.reproject_match(vi_min['NDVI'])

        vis = {
            "NDVI": ndv(bands["nir"], bands["red"]),
            "EVI": evi(bands["nir"], bands["red"], bands["blue"]),
            "NDMI": ndm(bands["nir"], bands["swir"]),
            "CIRE": cire(bands["nir"], bands["rededge"]),
            "CCI": cci(bands["green"], bands["red"])
        }

        # Normalize using annual min/max
        normalized = {}
        for k in vis:
            # Reproject vis[k] to match vi_min[k]
            vi_matched = vis[k].rio.reproject_match(vi_min[k]).chunk({'x': 1024, 'y': 1024})

            # Apply the valid mask to the current VI
            vi_matched = vi_matched.where(valid_aligned == 1, np.nan)

            # Normalize by min and max
            normalized[k] = (vi_matched - vi_min[k]) / (vi_max[k] - vi_min[k])

            # Mask invalid pixels and infinities
            normalized[k] = normalized[k].where(np.isfinite(normalized[k]), np.nan)

            # vi_matched.transpose('band', 'y', 'x').rio.to_raster(f'Test_{k}.tif')

        return normalized


def pull_closest_from_stac(target_date_str, stac_loc='https://data.geo.admin.ch/api/stac/v0.9/',
                           collection='ch.swisstopo.swisseo_s2-sr_v100'):
    """
    Connect to the swisstopo STAC and find the item closest to the target date.

    :param target_date_str: string in 'YYYY-MM-DD' format
    :param stac_loc: STAC endpoint
    :param collection: collection to search
    :return: STAC Item closest to the target date
    """
    target_date = datetime.strptime(target_date_str, "%Y-%m-%d")

    # Connect to STAC
    service = pystac_client.Client.open(stac_loc)
    service.add_conforms_to("COLLECTIONS")
    service.add_conforms_to("ITEM_SEARCH")

    # Search items for the year of the target date
    start_date = f'{target_date.year}-04-01'
    end_date = f'{target_date.year}-09-30'
    item_search = service.search(collections=[collection], datetime=f'{start_date}/{end_date}')
    item_list = list(item_search.items())

    if not item_list:
        print(f"No items found in {target_date.year}")
        return None

    # Find the closest item using your get_item_datetime function
    closest_item = min(item_list, key=lambda item: abs(get_item_datetime(item) - target_date))

    print(f"Closest image to {target_date_str} is {closest_item.id} ({get_item_datetime(closest_item).date()})")
    return closest_item


if __name__ == '__main__':
    pass
