from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.cm as cm
import imageio.v3 as iio
import rioxarray as rxr
import contextily as cx
import geopandas as gpd
from PIL import Image
import seaborn as sns
import pandas as pd
import numpy as np
import calendar
import os
import re


def get_ordinal(n):
    """Returns the ordinal suffix for an integer (e.g., 1st, 2nd, 3rd)."""
    if 11 <= n <= 13:
        return f"{n}th"
    else:
        suffixes = {1: 'st', 2: 'nd', 3: 'rd'}
        return f"{n}{suffixes.get(n % 10, 'th')}"


def create_example_grid(folder_path, output_name="example_grid.png", start_date=None, end_date=None, crop_ratio=0.15,
                        top_crop=0.05):
    """
    Create a grid of example plots
    :param folder_path:
    :param output_name:
    :param start_date:
    :param end_date:
    :param crop_ratio:
    :param top_crop:
    :return:
    """
    pattern = re.compile(r"frame_(\d{4})-(\d{2})-(\d{2})\.png$")
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date() if start_date else None
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date() if end_date else None

    temp_data = {i: [] for i in range(1, 13)}

    for file in sorted(os.listdir(folder_path)):
        match = pattern.search(file)
        if match:
            y, m, d = map(int, match.groups())
            current_dt = datetime(y, m, d).date()
            if (start_dt and current_dt < start_dt) or (end_dt and current_dt > end_dt):
                continue
            temp_data[m].append((os.path.join(folder_path, file), get_ordinal(d)))

    active_months = [m for m in range(1, 13) if temp_data[m]]
    if not active_months: return

    max_rows = len(max(temp_data.values(), key=len))
    num_cols = len(active_months)

    # 1. Determine Dimensions
    sample_path = temp_data[active_months[0]][0][0]
    with Image.open(sample_path) as img:
        orig_w, orig_h = img.size
        crop_x = int(orig_w * (1 - crop_ratio))
        crop_y_top = int(orig_h * top_crop)
        # Calculate aspect ratio of the MAP part
        map_w = crop_x
        map_h = orig_h - crop_y_top
        map_aspect = map_h / map_w

    # 2. Create Plot - Adjusting figsize to be very tall
    fig, axes = plt.subplots(max_rows, num_cols,
                             figsize=(num_cols * 3, max_rows * 3 * map_aspect),
                             squeeze=False)

    last_img_full = None

    for col_idx, month_num in enumerate(active_months):
        day_list = temp_data[month_num]
        for row_idx in range(max_rows):
            ax = axes[row_idx, col_idx]
            ax.axis('off')

            if row_idx < len(day_list):
                img_path, day_text = day_list[row_idx]
                img = Image.open(img_path)
                map_crop = img.crop((0, crop_y_top, crop_x, orig_h))
                ax.imshow(map_crop, aspect='equal')
                last_img_full = img

                ax.text(0.95, 0.95, day_text, transform=ax.transAxes,
                        ha='right', va='top', fontsize=14)
                        # bbox=dict(facecolor='white', alpha=0.7, lw=0, pad=1))

            if row_idx == 0:
                ax.set_title(calendar.month_name[month_num],
                             fontsize=16, pad=10)

    # Final layout adjustments
    plt.subplots_adjust(wspace=0.03, hspace=0.04, left=0.05, right=0.90, top=0.92, bottom=0.08)

    plt.savefig(output_name, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Grid saved to: {output_name}")


def create_climate_grid(folder_path, output_name="precip_grid.png", crop_ratio=0.15, top_crop=0.05):
    """
    Creates a grid with month names, removes plot artifacts from the color bar,
    and ensures the color bar scale is not stretched.
    """
    pattern = re.compile(r"(\d{4})_(\d{2})\.png$")
    data = {}
    for file in os.listdir(folder_path):
        match = pattern.search(file)
        if match:
            year, month = match.groups()
            data.setdefault(year, {})[month] = os.path.join(folder_path, file)

    if not data:
        print("No matching files found.")
        return

    years = sorted(data.keys())
    months = sorted(list(set(m for y in data for m in data[y])))

    # 1. Determine dimensions
    first_path = data[years[0]][months[0]]
    with Image.open(first_path) as img:
        w, h = img.size
        crop_x = int(w * (1 - crop_ratio))
        crop_y_top = int(h * top_crop)
        # Calculate aspect ratio of the cropped image to set figsize correctly
        crop_h = h - crop_y_top
        aspect = crop_h / crop_x

    # Increase figsize and use a tighter hspace
    fig, axes = plt.subplots(len(years), len(months),
                             figsize=(len(months) * 4, len(years) * 4 * aspect))

    if len(years) == 1: axes = axes[None, :]
    if len(months) == 1: axes = axes[:, None]

    for r, year in enumerate(years):
        for c, month in enumerate(months):
            ax = axes[r, c]
            ax.axis('off')

            if month in data[year]:
                img = Image.open(data[year][month])
                map_crop = img.crop((0, crop_y_top, crop_x, h))

                # 'aspect='auto' allows the image to fill the subplot space
                # If your images MUST be equal aspect, keep 'equal' but
                # ensure your figsize matches the image dimensions perfectly.
                ax.imshow(map_crop, aspect='equal')
                last_img_full = img

            # --- Enlarged Text ---
            if r == 0:
                month_name = calendar.month_name[int(month)]
                ax.text(0.5, 1.05, month_name, transform=ax.transAxes,
                        ha='center', fontsize=20, fontweight='bold')

            if c == 0:
                ax.text(-0.15, 0.5, year, transform=ax.transAxes, rotation=0,
                        ha='right', va='center', fontsize=20, fontweight='bold')

    # 2. Add the Color Bar
    if last_img_full:
        cbar_start_x = crop_x + 5
        cbar_img = last_img_full.crop((cbar_start_x, crop_y_top, w, h))

        # Adjusted for larger text/layout
        cax = fig.add_axes([0.92, 0.1, 0.04, 0.8])
        cax.imshow(cbar_img, aspect='equal')
        cax.axis('off')

    # --- THE KEY TO CLOSING GAPS ---
    # Manually force the spacing. hspace=0.0 means no space between rows.
    plt.subplots_adjust(wspace=0.02, hspace=0.02, left=0.1, right=0.9, top=0.9, bottom=0.1)

    plt.savefig(output_name, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Grid saved to: {output_name}")


def plot_national_summary(data, output_location=None):

    # Calculate the percentage on the fly
    data['percent'] = (data['Sum_Threshold_Passed'] / data['Total_Non_NaN_Cells']) * 100

    # 2. Map integer months (5, 6, 7...) to names ("May", "June"...)
    # This aligns your dataframe with your month_order list
    month_name_map = {i: calendar.month_name[i] for i in range(1, 13)}
    data["month_name"] = data["month"].map(month_name_map)

    # Global plotting font size
    plt.rcParams.update({'font.size': 14})
    sns.set_style("whitegrid")

    # Ensure the months are ordered correctly
    month_order = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]

    # 3. Filter to only months present in data using the new names
    unique_months = [m for m in month_order if m in data["month_name"].unique()]
    data["month_name"] = pd.Categorical(data["month_name"], categories=unique_months, ordered=True)

    # Plot dimensions
    plot_height = 4
    plot_aspect = 1.2

    # 4. Create the FacetGrid using month_name
    g1 = sns.FacetGrid(data, col="month_name", col_order=unique_months,
                       margin_titles=False, height=plot_height, aspect=plot_aspect)

    # Map the custom plotting function
    # Ensure national_plot_style expects 'percent' and 'day' columns
    g1.map_dataframe(national_plot_style)

    # Calculate global Y limits
    max_val = data['percent'].max()
    y_max = max_val * 1.1 if max_val > 0 else 10
    y_min = 0

    # Adjust each subplot
    for ax, title in zip(g1.axes.flat, unique_months):
        ax.set_ylim(y_min, y_max)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
        ax.set_title(title)
        ax.yaxis.grid(True, linestyle="--", alpha=0.6)
        ax.xaxis.grid(False)

    g1.set_axis_labels("", "Percent of European Beech\nArea Discolored")
    plt.tight_layout()

    # Save logic
    if output_location:
        if not os.path.exists(output_location):
            os.makedirs(output_location)
        figure_name = 'National_Summary_Percent_Discoloration.png'
        plt.savefig(os.path.join(output_location, figure_name), dpi=300)

    plt.show()
    plt.close(g1.fig)


def national_plot_style(data, **kwargs):
    """
    Replicates the bar + dashed line look.
    **kwargs catches 'color', 'label', etc. passed by FacetGrid.
    """
    ax = plt.gca()

    # Ensure years are sorted for this specific month
    data = data.sort_values("year")

    # Map years to numeric positions for consistent spacing
    unique_years = np.sort(data["year"].unique())
    year_pos_map = {year: i for i, year in enumerate(unique_years)}
    subset_x = [year_pos_map[y] for y in data["year"]]

    # 1. Plot Bars ['#0A2F1F', '#228B22', '#9ACD32', '#FFFF00', '#FFFFE0']
    # We use a hardcoded color, but kwargs['color'] would contain
    # the palette color if you wanted to use Seaborn's colors.
    ax.bar(subset_x, data["percent"], width=0.4, alpha=0.8, color='k')

    # 2. Add the connecting line
    # ax.plot(subset_x, data["percent"], '--', color='black', alpha=0.2, linewidth=1.5)

    # 3. Add markers
    # ax.scatter(subset_x, data["percent"], marker='_', color='black', alpha=0.6, s=40, linewidths=1.25)

    # 4. Set x-ticks to match the years
    ax.set_xticks(range(len(unique_years)))
    ax.set_xticklabels(unique_years)


def plot_district_summary(data, shapefile, id_field='District', output=None, annotate=False):
    """
    Plots a tight grid of maps (3 columns for June, July, August)
    with a 0-10 color stretch.
    """
    df = pd.read_csv(data)
    df = df[df.Total_Non_NaN_Cells > 1000]

    # Calculate percentage (0-100 scale)
    df['percent'] = (df['Sum_Threshold_Passed'] / df['Total_Non_NaN_Cells']) * 100

    # Load and prep districts
    districts = gpd.read_file(shapefile)
    districts['district_link'] = districts[id_field]

    # Define Grid Dimensions
    years = [2017, 2018, 2019, 2020, 2021, 2022, 2023]
    months = [6, 7, 8]  # ['June', 'July', 'August']

    # Define Colormap and 0-10 Normalization
    colors = ['#0A2F1F', '#228B22', '#5EAC2A', '#9ACD32', '#FFFF00']
    # ['#0A2F1F', '#228B22', '#9ACD32', '#FFFF00', '#FFFFE0']  # Updated for Colorblind Friendliness
    # LinearSegmented creates the "stretched" gradient effect
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_stretch", colors)
    norm = mcolors.Normalize(vmin=0, vmax=10)

    # Setup Plot Grid
    # Reducing figsize and using sharex/y helps maintain alignment
    fig, axes = plt.subplots(nrows=len(years), ncols=len(months),
                             figsize=(10, 1.8 * len(years)),
                             sharex=True, sharey=True)

    for r, year in enumerate(years):
        for c, month in enumerate(months):
            ax = axes[r, c]

            # Filter data
            date_data = df[(df['year'] == year) & (df['month'] == month)].copy()

            # Plot background
            districts.plot(ax=ax, color='#eeeeee', edgecolor='white', linewidth=0.2)

            if not date_data.empty:

                map_data = districts.merge(date_data, left_on='district_link', right_on='District', how='left')

                map_data.plot(
                    column='percent',
                    ax=ax,
                    cmap=cmap,
                    norm=norm,
                    edgecolor='black',
                    linewidth=0.05
                )

                if annotate:
                    for _, row in map_data.iterrows():
                        if row.geometry and not pd.isna(row['percent']):
                            centroid = row.geometry.centroid
                            ax.text(centroid.x, centroid.y, f"{row['percent']:.0f}",
                                    fontsize=6, ha='center', color='grey', alpha=0.6)

            # Column Titles (Months)
            if r == 0:
                ax.set_title(month, fontsize=14, fontweight='bold', pad=5)

            # Row Labels (Years)
            if c == 0:
                # Placed as text to avoid layout shifting from y-labels
                ax.text(-0.2, 0.5, str(year), transform=ax.transAxes,
                        fontsize=14, fontweight='bold', va='center', ha='right')

            ax.set_axis_off()

    # Tighten the layout
    plt.subplots_adjust(left=0.1, right=0.9, top=0.92, bottom=0.05, wspace=0.02, hspace=0.01)

    # Global colorbar
    cbar_ax = fig.add_axes([0.93, 0.25, 0.015, 0.5])
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Percent Discoloration (%)', fontsize=10)

    if output:
        plt.savefig(output, dpi=300, bbox_inches='tight', transparent=True)

    plt.show()


def get_ordinal(n):
    if 10 <= n % 100 <= 20:
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f"{n}{suffix}"


def format_date(date_str):
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    day = get_ordinal(date_obj.day)
    return date_obj.strftime(f"%B {day} %Y")


def plot_example(data_loc, extent_shp, dpi, output, color_ramp=None):
    # Default to your requested colors if none provided
    if color_ramp is None:
        color_ramp = ['#0A2F1F', '#228B22', '#9ACD32', '#FFFF00', '#FFFFE0']

    # Get date from file name
    date_str = os.path.basename(data_loc).split("_")[-1].replace(".tif", "")

    # Open with 'mask=True' to handle NoData immediately
    ds = rxr.open_rasterio(data_loc, mask=True).squeeze()

    # Match CRS of shapefile to Raster before clipping
    gdf = gpd.read_file(extent_shp).to_crs(ds.rio.crs)
    ds_clipped = ds.rio.clip(gdf.geometry.values, ds.rio.crs)

    # Scale the raster
    scale_factor = ds_clipped.attrs.get('scale_factor', 1.0)
    add_offset = ds_clipped.attrs.get('add_offset', 0.0)

    if scale_factor != 1.0 or add_offset != 0.0:
        ds_scaled = (ds_clipped * scale_factor) + add_offset
    else:
        ds_scaled = ds_clipped

    # Reproject to match Contextily
    ds_web = ds_scaled.rio.reproject("EPSG:3857")

    # Cleanup outliers
    final_ds = ds_web.where((ds_web >= 0) & (ds_web <= 10000), np.nan)

    if final_ds.isnull().all():
        print(f"Skipping {date_str}: Data is all NaN after filtering 0-1.")
        # Debug: Print the range of ds_scaled to see why it's being filtered out
        print(f"Debug Info: Scaled Min: {ds_scaled.min().values}, Max: {ds_scaled.max().values}")
        return

    # 4. Plot
    fig, ax = plt.subplots(figsize=(13, 8))

    cmap = LinearSegmentedColormap.from_list("custom_stretch", color_ramp)
    cmap.set_bad(alpha=0)  # Ensure NoData is fully transparent

    # Plot the probability map
    im = final_ds.plot.imshow(
        ax=ax,
        cmap=cmap,
        vmin=0,
        vmax=1,
        alpha=0.9,  # Balanced to see satellite detail below
        add_colorbar=False,
        zorder=2
    )

    # 5. Add Satellite Basemap
    try:
        cx.add_basemap(
            ax,
            crs="EPSG:3857",
            source=cx.providers.Esri.WorldImagery,  # WorldTopoMap
            alpha=0.65,
            zorder=1,
            zoom='auto'
        )
    except Exception as e:
        print(f"Basemap Error: {e}")

    # 6. Formatting & Colorbar
    ax.set_title(format_date(date_str), fontsize=15, fontweight='bold')
    ax.axis('off')

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, aspect=12)
    cbar.ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    cbar.set_label('Discoloration Probability', fontsize=10)

    # 7. Save
    out_file = os.path.join(output, f"frame_{date_str}.png")
    plt.savefig(out_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def process_examples(data_folder, extent_loc, dpi, output, color_ramp=None):
    # Removed background_loc from arguments
    if not os.path.exists(output):
        os.makedirs(output)

    data_to_process = [f for f in os.listdir(data_folder) if f.lower().endswith(".tif")]

    for tif_name in data_to_process:
        tif_path = os.path.join(data_folder, tif_name)
        print(f'Processing {tif_name}')
        plot_example(tif_path, extent_loc, dpi, output, color_ramp)


def make_video_or_gif(folder_loc, fps, output, start_date=None, end_date=None):
    """
    folder_loc: Directory containing the .png frames
    fps: Frames per second
    output: Full path for output (e.g., 'animation.gif' or 'animation.mp4')
    start_date: String 'YYYY-MM-DD' (optional)
    end_date: String 'YYYY-MM-DD' (optional)
    """
    # 1. Gather and Filter Files
    all_files = [f for f in os.listdir(folder_loc) if f.lower().endswith(".png")]
    start_dt = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
    end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None

    valid_frames = []
    for f in all_files:
        try:
            # Assumes format: "any_text_YYYY-MM-DD.png"
            date_part = f.split("_")[-1].replace(".png", "")
            file_dt = datetime.strptime(date_part, "%Y-%m-%d")

            if start_dt and file_dt < start_dt: continue
            if end_dt and file_dt > end_dt: continue

            valid_frames.append((file_dt, os.path.join(folder_loc, f)))
        except (ValueError, IndexError):
            continue

    valid_frames.sort(key=lambda x: x[0])
    if not valid_frames:
        print("No frames found. Check your date range or filenames.")
        return

    # 2. Load Frames
    print(f"Processing {len(valid_frames)} frames...")
    frames = [iio.imread(path) for _, path in valid_frames]

    # 3. Handle Output Type
    ext = os.path.splitext(output)[1].lower()

    if ext == '.mp4':
        # High-quality video (prevents color pixelation)
        iio.imwrite(
            output,
            frames,
            fps=fps,
            codec='libx264',
            quality=9,  # 0-10 scale
            pixelformat='yuv420p',  # Ensures compatibility with most players
            macro_block_size=None
        )
    else:
        # Optimized GIF (uses better color quantization)
        iio.imwrite(
            output,
            frames,
            fps=fps,
            loop=0,          # 0 means infinite loop
            quantizer='nq'   # NeuQuant reduces gradient "banding"
        )

    print(f"Success! Saved to: {output}")


def plot_correlation(data, columns, output=None):
    """
    Plots a Pearson correlation heatmap for specified columns in a triangle shape
    with enlarged text for readability.
    """

    df = pd.read_csv(data)

    # Rename columns as requested
    df = df.rename(columns={'CRE': 'ClRE', 'NDM': 'NDWI', 'NDV': 'NDVI'})

    # Calculate correlation and create mask for the upper triangle
    corr = df[columns].corr(method='pearson')
    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=(12, 10)) # Slightly larger figure size

    # sns.heatmap adjustments:
    # annot_kws sets the size of the numbers inside the squares
    ax = sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap='coolwarm',
        vmin=0,
        vmax=1,
        center=0.5,
        square=True,
        linewidths=.5,
        annot_kws={"size": 16},  # Larger correlation numbers
        cbar_kws={"shrink": .8}
    )

    # Increase the size of the axis labels (variable names)
    plt.xticks(fontsize=16, rotation=45, ha='right')
    plt.yticks(fontsize=16, rotation=0)

    # Increase the size of the colorbar tick labels
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)

    # If you decide to use a title, set it large here
    # plt.title('Pearson Correlation of Vegetation Indices', fontsize=20, pad=20)

    plt.tight_layout()

    if output:
        plt.savefig(output)

    plt.close()


def plot_temporal_distribution(data, output=None):
    """
    Plots a categorical histogram showing percentages with
    custom labels for the 'disco' legend, with enlarged text
    and removed x-axis labels.
    """
    df = pd.read_csv(data)
    df['Date'] = pd.to_datetime(df['Date'])

    # 1. Map the numeric values to descriptive strings for the legend
    disco_map = {1.0: 'Discolored', 0.0: 'Not Discolored', 1: 'Discolored', 0: 'Not Discolored'}
    df['Status'] = df['disco'].map(disco_map)

    # 2. Create 'Year-Month' string and sort chronologically
    df['YearMonth'] = df['Date'].dt.strftime('%Y-%m')
    df = df.sort_values('Date')

    # Increased figure size slightly to better fit larger text
    plt.figure(figsize=(16, 9))

    # 3. Plot using the new 'Status' column for the hue
    ax = sns.histplot(
        data=df,
        x='YearMonth',
        hue='Status',
        multiple='dodge',
        stat='density',
        kde=False,
        shrink=0.8,
        alpha=0.6,
        common_norm=False,
    )

    # Increase legend size
    sns.move_legend(
        ax,
        loc="upper right",
        title=None,
        frameon=False,
        fontsize=18,  # Significantly larger legend text
        labelspacing=0.6
    )

    # Make all other text larger
    plt.title('Percentage of Points by Active Months', fontsize=22, pad=20)
    plt.ylabel('Density', fontsize=20)

    # Remove the x-axis label (YearMonth)
    ax.set_xlabel('')

    # Increase size of the tick labels (the dates and density numbers)
    plt.xticks(rotation=45, ha='right', fontsize=16)
    plt.yticks(fontsize=16)

    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()

    if output:
        plt.savefig(output)

    plt.close()


if __name__ == '__main__':
    location = 'Hardwald'
    dataset = "DiscoCH_rbf_10"  # "DiscoCH_rbf_10_20_30"
    folder = r"F:\cb_overflow\2026_09_01_FORCE_Backup\FORCE\output\2026\DiscoCH_2026_08_14_CCI" # fr"F:\cb_overflow\2026_09_02_FORCE\FORCE\{dataset}\output\level5norm\2026\mosaic\CCI" # r"B:\bloomc\DiscoCH_2026_08_03\FORCE\FORCE\output\2026\DiscoCH_2026_08_14"
    # 'B:\bloomc\DiscoCH_2026_08_03\FORCE\output\DiscoCH_FORCE_2026'
    shapefile_ = f'../data/example_extents/Bounding_Box_{location}.shp'
    output_ = fr"F:\cb_overflow\2026_09_02_FORCE\FORCE\{dataset}\output\level5norm\2026\examples\CCI\tiles\{location}"
    output_video = fr"F:\cb_overflow\2026_09_02_FORCE\FORCE\{dataset}\output\level5norm\2026\examples\CCI\gif\CCI_timelapse_{location}.gif"
    dpi_ = 300
    fps_ = 1
    nodata_val = -32768 # 65535
    colors = ['#0A2F1F', '#228B22', '#5EAC2A', '#9ACD32', '#FFFF00'] # ['#3A4F41', '#607D6B', '#9DB28C', '#D1C78D', '#E6D385']

    process_examples(data_folder=folder, extent_loc=shapefile_, dpi=dpi_, output=output_, color_ramp=colors)
    make_video_or_gif(output_, fps_, output_video, start_date='2026-06-01', end_date='2026-09-02')