import os
import gc
import json
import time
import pystac_client
from datetime import datetime, UTC
import xarray as xr
import rioxarray as rxr
import warnings
import numpy as np
import shutil
import dask
import matplotlib.pyplot as plt
import glob
import math
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
    return (rededge3 / rededge1) - 1  # rededge3: B07, rededge1: B05


def cci(green, red):
    return (green - red) / (green + red)  # green: B03, red: B04


def print_metadata(item):
    """
    Prints asset keys and band info directly from the STAC Item in v200.
    """
    print(f"--- STAC Item Assets for {item.id} ---")

    bands_10m = []
    bands_20m = []
    masks = []

    for key, asset in item.assets.items():
        if key.endswith("_10m.tif"):
            if "mask" in key:
                masks.append(key)
            else:
                bands_10m.append(key)
        elif key.endswith("_20m.tif"):
            bands_20m.append(key)

    print("10m Bands/Assets:")
    for b in sorted(bands_10m):
        print(f"  - {b}")

    print("\n20m Bands/Assets:")
    for b in sorted(bands_20m):
        print(f"  - {b}")

    print("\nMasks (10m):")
    for m in sorted(masks):
        print(f"  - {m}")


def pull_from_stac(stac_loc='https://data.geo.admin.ch/api/stac/v0.9/', year=2018, end_date=None, start_date=None):
    """
    Connect to the swisstopo stac and collect datasets
    :param start_date:
    :param end_date:
    :param stac_loc:
    :param year:
    :return:
    """
    # Create a connection with the STAC data collection
    service = pystac_client.Client.open(stac_loc)
    service.add_conforms_to("COLLECTIONS")
    service.add_conforms_to("ITEM_SEARCH")

    if end_date is None:
        end_date = f'{year}-09-30'
    if start_date is None:
        start_date = f'{year}-03-01'

    # Filter by start and end date - Updated to v200
    item_search = service.search(collections=['ch.swisstopo.swisseo_s2-sr_v200'], datetime=f'{start_date}/{end_date}')

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

    if folder is None:
        return None, None
    else:
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


def new_image_check(start_date, end_date, existing_data_loc, stac_loc='https://data.geo.admin.ch/api/stac/v0.9/'):
    """
    Checks for new data in the year of interest
    :param end_date:
    :param start_date:
    :param existing_data_loc:
    :param stac_loc:
    :return:
    """

    # Get year from date of interest
    year = int(start_date.split("-")[0])

    # Pull the list of available data
    items = pull_from_stac(stac_loc=stac_loc, year=year, end_date=end_date, start_date=start_date)

    # Open processed data
    if existing_data_loc is not None:
        existing_meta = load_minmax_metadata(year, existing_data_loc)
    else:
        existing_meta = None

    processed_dates = set(existing_meta["processed_dates"]) if existing_meta else set()

    # Find NEW items only
    new_items = [item for item in items if get_item_datetime(item).isoformat() not in processed_dates]

    print(f"Already processed: {len(processed_dates)}")
    print(f"New images to process: {len(new_items)}")

    if len(new_items) == 0:
        return None, None, None
    else:
        return new_items


def load_and_process_assets(item, forest_mask, bbox=None, band_metadata=False, verbose=0):
    """
    Load bands from a STAC item, align 20m → 10m, and return selected bands and valid mask.

    :param verbose:
    :param bbox: (minx, miny, maxx, maxy)
    :param forest_mask:
    :param band_metadata:
    :param item: A STAC Item (its .assets dict is used to locate the band files)
    :return: dict of bands, valid_mask (xr.DataArray)
    """
    tick = time.time()
    key_problem = None
    assets = item.assets

    try:
        # Get individual v200 asset keys
        b02_key = next(k for k in assets.keys() if k.endswith("_b02_10m.tif"))
        b03_key = next(k for k in assets.keys() if k.endswith("_b03_10m.tif"))
        b04_key = next(k for k in assets.keys() if k.endswith("_b04_10m.tif"))
        b08_key = next(k for k in assets.keys() if k.endswith("_b08_10m.tif"))
        b11_key = next(k for k in assets.keys() if k.endswith("_b11_20m.tif"))
        b05_key = next(k for k in assets.keys() if k.endswith("_b05_20m.tif"))
        b07_key = next(k for k in assets.keys() if k.endswith("_b07_20m.tif"))
        tmask_key = next(k for k in assets.keys() if k.endswith("_terrainmask_10m.tif"))
        cmask_key = next(k for k in assets.keys() if k.endswith("_cloudmask_10m.tif"))
    except StopIteration as e:
        key_problem = 1
        print(f"  WARNING: Problem loading the v200 single-band assets within this item! Skipping. {e}")

    if key_problem is not None:
        return None, None
    else:
        # Optionally print out the various asset bands
        if band_metadata:
            print_metadata(item)

        # Load rasters lazily as dask chunks
        red = rxr.open_rasterio(assets[b04_key].href, masked=True, chunks={"x": 1024, "y": 1024})
        green = rxr.open_rasterio(assets[b03_key].href, masked=True, chunks={"x": 1024, "y": 1024})
        blue = rxr.open_rasterio(assets[b02_key].href, masked=True, chunks={"x": 1024, "y": 1024})
        nir = rxr.open_rasterio(assets[b08_key].href, masked=True, chunks={"x": 1024, "y": 1024})
        swir = rxr.open_rasterio(assets[b11_key].href, masked=True, chunks={"x": 1024, "y": 1024})
        rededge1 = rxr.open_rasterio(assets[b05_key].href, masked=True, chunks={"x": 1024, "y": 1024})
        rededge3 = rxr.open_rasterio(assets[b07_key].href, masked=True, chunks={"x": 1024, "y": 1024})

        terrain_mask = rxr.open_rasterio(assets[tmask_key].href, masked=True, chunks={"x": 1024, "y": 1024})
        cloud_mask = rxr.open_rasterio(assets[cmask_key].href, masked=True, chunks={"x": 1024, "y": 1024})

        # Apply spatial subset
        if bbox is not None:
            try:
                red = red.rio.clip_box(*bbox)
                green = green.rio.clip_box(*bbox)
                blue = blue.rio.clip_box(*bbox)
                nir = nir.rio.clip_box(*bbox)
                swir = swir.rio.clip_box(*bbox)
                rededge1 = rededge1.rio.clip_box(*bbox)
                rededge3 = rededge3.rio.clip_box(*bbox)
                terrain_mask = terrain_mask.rio.clip_box(*bbox)
                cloud_mask = cloud_mask.rio.clip_box(*bbox)
            except NoDataInBounds:
                print("  No data in bbox, skipping asset")
                return None, None

        if verbose == 1:
            load_time = time.time()
            print(f'  Loaded Data and Masks in {load_time - tick:.2f} seconds')

        # Convert to float32 to save memory
        red = red.astype("float32", copy=False)
        green = green.astype("float32", copy=False)
        blue = blue.astype("float32", copy=False)
        nir = nir.astype("float32", copy=False)

        if verbose == 1:
            astype10_time = time.time()
            print(f'  Converted 10m to float32 in {astype10_time - load_time:.2f} seconds')

        # Reproject Match 20m bands (theoretically lazy, but it seems to compute partially)
        swir = swir.rio.reproject_match(red, dtype="float32")
        rededge1 = rededge1.rio.reproject_match(red, dtype="float32")
        rededge3 = rededge3.rio.reproject_match(red, dtype="float32")

        if verbose == 1:
            astype20_time = time.time()
            print(f'  Converted 20m to 10m and float32 in {astype20_time - astype10_time:.2f} seconds')

        # Create a combined terrain and cloud mask
        valid_mask = ((cloud_mask != 1) & (terrain_mask <= 63) & (red > 0) & (forest_mask == 1))

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
            "rededge1": rededge1,
            "rededge3": rededge3
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

        # Clean memory
        del red, green, blue, nir, swir, rededge1, rededge3, terrain_mask, cloud_mask
        gc.collect()

    return bands, valid_mask


def build_template(national_template, bbox, output=False):
    """
    Build a nodata template raster for a given bounding box
    using the provided national template. Writes a GeoTIFF
    next to the national template and returns the file path.
    """
    # Load the national template
    national_da = (
        rxr.open_rasterio(national_template)
        .squeeze(drop=True)
    )

    # Crop to the bounding box
    bbox_template = national_da.rio.clip_box(*bbox)

    # Convert 255 nodata values to real NaNs
    bbox_template = (
        bbox_template
        .astype("float32")
        .where(bbox_template != 255)
    )

    del national_da
    gc.collect()

    if output:
        # Build output path in same directory
        template_dir = os.path.dirname(national_template)
        template_name = os.path.splitext(os.path.basename(national_template))[0]

        out_path = os.path.join(
            template_dir,
            f"{template_name}_bbox_template.tif"
        )

        # Write GeoTIFF
        bbox_template.rio.to_raster(out_path)

    return bbox_template


def plot_disco_result(raster, item_date, output_dir, min_pixel_count=100, show=True):
    # Import lazily so environments without a working contextily/geopy
    # install (or with broken SSL cert stores) can still run the pipeline
    # with plotting disabled.
    import contextily as cx

    # 1. Load data
    if isinstance(raster, str):
        da = xr.open_dataarray(raster).squeeze()
    elif isinstance(raster, xr.DataArray):
        da = raster.squeeze()
    else:
        raise TypeError(
            "raster must be a file path or an xarray.DataArray (rioxarray) to plot"
        )
    plot_data = da.where(da != -9999)

    # 2. Check density
    if int(plot_data.count()) < min_pixel_count:
        print("This asset does not have enough valid pixels to plot")
        return False

    # 3. Setup Colors
    disco_colors = ['#0A2F1F', '#228B22', '#9ACD32', '#FFFF00', '#FFFFE0']

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
        alpha=1,
        aspect='auto',

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
    if output_dir is not None:
        plt.savefig(os.path.join(output_dir, f"Disco_Proba_{item_date}.png"), dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    plt.close()


def plot_disco_grid(raster_dir, pattern="Disco_Proba_*.tif", cols=3):
    """
    Creates a grid of all discoloration maps overlaid on satellite imagery.
    Ensures correct aspect ratio (no stretching) and consistent scaling.
    """
    # Import lazily so environments without a working contextily/geopy
    # install (or with broken SSL cert stores) can still run the pipeline
    # with plotting disabled.
    import contextily as cx

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


def update_vi_min_max(items_to_process, year_of_interest, existing_data, forest_mask, template, bbox,
                      band_metadata=False, run_after_each_update=False, disco_model=None, output_dir=None,
                      plot_results=True):
    """
    Updates (or creates) VI min max rasters and metadata
    :param output_dir:
    :param disco_model:
    :param run_after_each_update:
    :param bbox:
    :param template:
    :param forest_mask:
    :param existing_data:
    :param items_to_process: List of items to process in the year of interest
    :param year_of_interest: Year to process
    :param band_metadata: Print out information from the STAC
    :param plot_results: If True (default), display/save the discoloration plot after each
        update when run_after_each_update and disco_model are set. Set False to skip plotting
        entirely (e.g. for headless batch runs) while still writing the output raster.
    :return:
    """

    # Initialize the vi_min, vi_max, and metadata
    vi_min = None
    vi_max = None
    existing_meta = None

    drains_forest_mask = rxr.open_rasterio(forest_mask)
    if bbox is not None:
        drains_forest_mask = drains_forest_mask.rio.clip_box(*bbox)

    # Iterate over the items available in the SwissEO STAC
    for i, item in enumerate(items_to_process):
        print(f"\n Processing {i + 1}/{len(items_to_process)} : {item.id}")

        if band_metadata:
            print_metadata(item)

        bands, valid = load_and_process_assets(item, drains_forest_mask, bbox, band_metadata=band_metadata)

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
                "CIRE": cire(bands["rededge3"], bands["rededge1"]),
                "CCI": cci(bands["green"], bands["red"])
            }

            vi_time = time.time()
            print(f'  Created VIs in {vi_time - tick:.2f} seconds')

            # Initialize on first image if no previous VI min max was loaded
            if vi_min is None:
                if existing_data is not None:
                    try:
                        vi_min, vi_max = load_minmax_rasters(year_of_interest, existing_data, chunks='default')
                    except FileNotFoundError:
                        vi_min, vi_max = None, None

                if vi_min is None:
                    print(f'  No existing VI min or max data was found. Creating new files.')

                    # Load the template raster with the full extent to capture all orbits
                    template_raster = template.squeeze(drop=True).astype("float32")

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

                if existing_data is not None:
                    # Compute and Save to disk
                    save_minmax_rasters({k: vi_min[k]}, {k: vi_max[k]}, year_of_interest, existing_data)

            # Free up memory between runs
            del vis
            gc.collect()

            comp_time = time.time()
            print(f'  Compared to existing Min Max in {comp_time - vi_time:.2f} seconds')

            # Revise the metadata with processed items
            if existing_data is not None:
                existing_meta = load_minmax_metadata(year_of_interest, existing_data)
            processed_dates = set(existing_meta["processed_dates"]) if existing_meta else set()
            updated_dates = sorted(set(processed_dates) | {get_item_datetime(item).isoformat()})
            if existing_data is not None:
                save_minmax_metadata(year_of_interest, updated_dates, existing_data)
            else:
                existing_meta = {
                    "year": year_of_interest,
                    "processed_dates": updated_dates,
                    "last_updated": datetime.now(UTC).isoformat()
                }

            backup_time = time.time()
            print(f"  Min/Max saved + metadata updated in {backup_time - comp_time:.2f} seconds")

            if run_after_each_update:
                print("  Running normalization and model application")

                # # Reload fresh min max rasters
                # vi_min_final, vi_max_final = load_minmax_rasters(
                #     year_of_interest, existing_data
                # )

                # Normalize only the current item
                normalized_vis = normalize_vis(
                    item,
                    vi_min,
                    vi_max,
                    forest_mask,
                    bbox
                )

                item_date = get_item_datetime(item).date().isoformat()

                disco_out = None

                # Build output file name for discoloration model or save normalized vis
                if output_dir is not None:

                    # Export VIs if no discoloration model is passed
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

                # Apply the discoloration model
                if disco_model is not None:
                    # Capture the returned DataArray for immediate plotting
                    disco_da = apply_disco(normalized_vis, disco_model, disco_out)

                    # Plotting is optional: controlled by plot_results
                    if plot_results:
                        plot_disco_result(disco_da, item_date, output_dir, show=True)


def normalize_vis(closest_data, vi_min, vi_max, forest_mask, bounding):
    """
    Create and normalize vegetation indices by min and max
    :param bounding:
    :param forest_mask:
    :param closest_data:
    :param vi_min:
    :param vi_max:
    :return:
    """

    drains_forest_mask = rxr.open_rasterio(forest_mask)
    if bounding is not None:
        drains_forest_mask = drains_forest_mask.rio.clip_box(*bounding)

    bands, valid = load_and_process_assets(closest_data, drains_forest_mask, bounding)

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
            "CIRE": cire(bands["rededge3"], bands["rededge1"]),
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
                           collection='ch.swisstopo.swisseo_s2-sr_v200'):
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
    # Fetch STAC items for July 2026
    items = pull_from_stac(year=2026, start_date='2026-07-01', end_date='2026-07-15')

    if items:
        # Pass the first STAC item directly to the function
        print_metadata(items[0])
    else:
        print("No items found matching the search criteria.")