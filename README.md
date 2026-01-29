# DiscoCH

DiscoCH is a Python tool for applying WSL's DRAINS project discoloration model to Sentinel-2 timeseries.

---

## Example Usage

To run the example, load the Jupyter Notebook in the notebooks directory and update the input and output file paths in src/disco_ch/user_defined_params.py.
These input and output locations are where the code will look for and save data.
Several datasets are also required to run the code including a no-data template and forest mask. These datasets can be requested by emailing colin.bloom@wsl.ch


---

## Installation
Clone the repository and install the environment using the provided YAML in a terminal window:
```bash
# Clone the directory
git clone https://github.com/ckb1oom/DiscoCH

# Navigate into the directory
cd DiscoCH

# Create an environment from the provided yaml file
conda env create -f environment.yml

# Install the package in editable mode
pip install -e .
```
Note: Automatic environment creation from the YAML has been problematic for some users. If you encounter issues, you can create the environment manually:
```bash
# Clone the directory
git clone https://github.com/ckb1oom/DiscoCH

# Navigate into the directory
cd DiscoCH

# Create and activate a new Conda environment
conda create -n DiscoCH python=3.11 -c conda-forge
conda activate DiscoCH

# Install core dependencies
conda install -c conda-forge rioxarray dask jupyterlab matplotlib pandas contextily scikit-learn=1.4.0
pip install pystac-client

# Install the package in editable mode
pip install -e .
```