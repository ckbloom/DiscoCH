import numpy as np
import xarray as xr
import pickle


def apply_disco(vis, model_loc, output):
    """
    Apply pickled ML model using rioxarray throughout, maintaining feature names.
    """

    # 1. Load model
    with open(model_loc, "rb") as f:
        model = pickle.load(f)

    # 2. Map input dictionary to a Dataset with specific model names
    # This ensures the features are named CCI, NDM, NDV, EVI, CRE
    ds = xr.Dataset({
        "Season_Peak_pct_CCI": vis["CCI"],
        "Season_Peak_pct_CRE": vis["CIRE"],
        "Season_Peak_pct_EVI": vis["EVI"],
        "Season_Peak_pct_NDM": vis["NDMI"],
        "Season_Peak_pct_NDV": vis["NDVI"]
    })

    # 3. Define explicit feature names
    feature_cols = [
        "Season_Peak_pct_CCI",
        "Season_Peak_pct_CRE",
        "Season_Peak_pct_EVI",
        "Season_Peak_pct_NDM",
        "Season_Peak_pct_NDV"
    ]

    # Convert and keep ONLY those columns
    full_df = ds.to_dataframe()
    df = full_df[feature_cols].dropna()

    # 4. Predict
    y_flat = np.full(ds.Season_Peak_pct_CCI.size, np.nan, dtype="float32")

    if not df.empty:
        predictions = model.predict(df)

        # Map predictions back using the index of the original dataframe
        # This ensures the NaNs stay in the right place spatially
        y_flat[full_df.index.get_indexer(df.index)] = predictions

    # 5. Reshape and Create rioxarray DataArray
    # We use the coordinates from the original CCI layer
    y_raster = y_flat.reshape(ds.Season_Peak_pct_CCI.shape)

    out_da = xr.DataArray(
        y_raster,
        dims=ds.Season_Peak_pct_CCI.dims,
        coords=ds.Season_Peak_pct_CCI.coords,
        name="disco"
    )

    # 6. Metadata and CRS
    # rioxarray uses the .rio accessor to handle geospatial metadata
    if ds.Season_Peak_pct_CCI.rio.crs:
        out_da.rio.write_crs(ds.Season_Peak_pct_CCI.rio.crs, inplace=True)

    # Standard GeoTIFFs expect a (band, y, x) structure
    out_da = out_da.expand_dims(dim="band", axis=0).assign_coords(band=[1])

    # 7. Write to Raster
    out_da.rio.to_raster(
        output,
        driver="GTiff",
        compress="deflate",
        dtype="float32",
        nodata=-9999
    )


# Run the code
if __name__ == '__main__':
    pass
