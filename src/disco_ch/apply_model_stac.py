import numpy as np
import xarray as xr
import pandas as pd
import pickle


def apply_disco(vis, model_loc, output):
    with open(model_loc, "rb") as f:
        model = pickle.load(f)

    feature_cols = [
        "Season_Peak_pct_CCI", "Season_Peak_pct_EVI",
        "Season_Peak_pct_CRE", "Season_Peak_pct_NDM",
        "Season_Peak_pct_NDV"
    ]

    def predict_wrapper(cci, evi, cre, ndm, ndv):
        features_array = np.stack([cci, evi, cre, ndm, ndv], axis=-1)
        orig_shape = features_array.shape[:-1]
        flat_features = features_array.reshape(-1, 5)

        df = pd.DataFrame(flat_features, columns=feature_cols)
        mask = df.notna().all(axis=1)

        # Initialize with NaN or your nodata value
        # We ensure this is float32 to hold decimal probabilities
        predictions = np.full(len(df), np.nan, dtype="float32")

        if mask.any():
            # .predict_proba() returns [n_samples, n_classes]
            # We take [:, 1] to get the probability of the positive class (1)
            probs = model.predict_proba(df[mask])
            predictions[mask] = probs[:, 1]

        return predictions.reshape(orig_shape)

    out_da = xr.apply_ufunc(
        predict_wrapper,
        vis["CCI"], vis["EVI"], vis["CIRE"], vis["NDMI"], vis["NDVI"],
        dask="allowed",
        output_dtypes=[np.float32]
    )

    # Standardize metadata and dimensions
    if "band" not in out_da.dims:
        out_da = out_da.expand_dims("band").assign_coords(band=[1])

    out_da = out_da.transpose("band", "y", "x")
    out_da.rio.write_crs(vis["CCI"].rio.crs, inplace=True)

    # Note: Use a standard float nodata like -9999 or NaN
    out_da.rio.to_raster(output, compress="deflate", nodata=-9999)

    return out_da

def apply_disco_dep(vis, model_loc, output):
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
        "Season_Peak_pct_EVI": vis["EVI"],
        "Season_Peak_pct_CRE": vis["CIRE"],
        "Season_Peak_pct_NDM": vis["NDMI"],
        "Season_Peak_pct_NDV": vis["NDVI"]
    })

    # 3. Define explicit feature names
    feature_cols = [
        "Season_Peak_pct_CCI",
        "Season_Peak_pct_EVI",
        "Season_Peak_pct_CRE",
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

    out_da = out_da.fillna(-9999)

    print(out_da)

    # 6. Metadata and CRS
    # rioxarray uses the .rio accessor to handle geospatial metadata
    if ds.Season_Peak_pct_CCI.rio.crs:
        out_da.rio.write_crs(ds.Season_Peak_pct_CCI.rio.crs, inplace=True)

    # Standard GeoTIFFs expect a (band, y, x) structure
    # out_da = out_da.assign_coords(band=[1])
    # if "band" not in out_da.dims:
    #     out_da = out_da.expand_dims(dim="band", axis=0).assign_coords(band=[1])

    out_da = out_da.squeeze("band", drop=True)
    # out_da = out_da.transpose("band", "y", "x")


    # 7. Write to Raster
    out_da.rio.to_raster(
        output,
        driver="GTiff",
        compress="deflate",
        dtype="float32",
        nodata=-9999
    )

def apply_disco_test(vis, model_loc, output):
    """
        Apply a pickled ML model to vegetation index DataArrays using xarray and rioxarray.
        Produces a GeoTIFF with the same spatial dimensions and CRS as the input.

        Args:
            vis (dict): Dictionary of xarray DataArrays with keys: "CCI", "CIRE", "EVI", "NDMI", "NDVI"
            model_loc (str): Path to the pickled sklearn model
            output (str): Output file path for the GeoTIFF
        """
    # -------------------------------
    # 1. Load the ML model
    # -------------------------------
    with open(model_loc, "rb") as f:
        model = pickle.load(f)

    # -------------------------------
    # 2. Map vis to a Dataset
    # -------------------------------
    ds = xr.Dataset({
        "Season_Peak_pct_CCI": vis["CCI"],
        "Season_Peak_pct_CRE": vis["CIRE"],
        "Season_Peak_pct_EVI": vis["EVI"],
        "Season_Peak_pct_NDM": vis["NDMI"],
        "Season_Peak_pct_NDV": vis["NDVI"]
    })

    # Explicit feature columns
    feature_cols = [
        "Season_Peak_pct_CCI",
        "Season_Peak_pct_CRE",
        "Season_Peak_pct_EVI",
        "Season_Peak_pct_NDM",
        "Season_Peak_pct_NDV"
    ]

    # -------------------------------
    # 3. Stack spatial dimensions
    # -------------------------------
    # Stack y/x into one dimension for sklearn prediction
    stacked = ds.stack(all_points=("y", "x"))

    # Create feature array in the correct order
    X = np.stack([stacked[var].values for var in feature_cols], axis=1)

    # -------------------------------
    # 4. Handle NaNs and predict
    # -------------------------------
    mask = ~np.any(np.isnan(X), axis=1)  # True for valid points
    y_pred = np.full(X.shape[0], np.nan, dtype="float32")

    if np.any(mask):
        # Reorder columns to match model feature names exactly
        # Map: your Dataset names -> model.feature_names_in_
        df_ordered = {name: stacked[var].values[mask]
                      for var, name in zip(feature_cols, model.feature_names_in_)}
        X_valid = np.column_stack([df_ordered[name] for name in model.feature_names_in_])

        # Predict
        y_pred[mask] = model.predict(X_valid)

    # -------------------------------
    # 5. Reshape back to original raster
    # -------------------------------
    y_raster = xr.DataArray(
        y_pred.reshape(ds.Season_Peak_pct_CCI.shape),
        dims=ds.Season_Peak_pct_CCI.dims,
        coords=ds.Season_Peak_pct_CCI.coords,
        name="disco"
    )

    # -------------------------------
    # 6. Replace NaNs with -9999 for nodata
    # -------------------------------
    y_raster = y_raster.fillna(-9999)

    # -------------------------------
    # 7. Add band dimension if missing
    # -------------------------------
    if "band" not in y_raster.dims:
        y_raster = y_raster.expand_dims(dim="band", axis=0).assign_coords(band=[1])

    # -------------------------------
    # 8. Ensure proper dimension order
    # -------------------------------
    y_raster = y_raster.transpose("band", "y", "x")

    # -------------------------------
    # 9. Set CRS
    # -------------------------------
    if ds.Season_Peak_pct_CCI.rio.crs:
        y_raster.rio.write_crs(ds.Season_Peak_pct_CCI.rio.crs, inplace=True)

    # -------------------------------
    # 10. Write to GeoTIFF
    # -------------------------------
    y_raster.rio.to_raster(
        output,
        driver="GTiff",
        compress="deflate",
        dtype="float32",
        nodata=-9999
    )

    return y_raster


# Run the code
if __name__ == '__main__':
    pass
