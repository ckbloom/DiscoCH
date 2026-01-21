# DiscoCH

DiscoCH is a Python tool for applying a discoloration model to Sentinel-2 timeseries.

---

## Example Usage

Load the Jupyter Notebook in the notebooks directory for a demonstration.

---

## Installation
Clone the repository and install the environment using the provided YAML:
```bash
git clone https://github.com/ckb1oom/DiscoCH
cd disco-ch
conda env create -f environment.yml
pip install -e .
```
Note: Automatic environment creation from the YAML has been problematic for some users. If you encounter issues, you can create the environment manually:
```bash
git clone https://github.com/ckb1oom/DiscoCH
cd DiscoCH

# Create and activate a new Conda environment
conda create -n DiscoCH python=3.11 -c conda-forge
conda activate DiscoCH

# Install core dependencies
conda install -c conda-forge rioxarray dask jupyterlab
pip install pystac-client

# Install the package in editable mode
pip install -e .

```