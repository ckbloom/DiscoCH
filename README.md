# DiscoCH

DiscoCH is a Python toolbox for applying WSL's DRAINS project discoloration model to Sentinel-2 timeseries.
The current model version corresponds with the model presented in
[Bloom et al., in Review](http://dx.doi.org/10.2139/ssrn.5343751) 
---
## Project Status

DiscoCH is a **research toolbox** and is under active development.

- This is an initial public release ('v0.0.1')
- Inputs and outputs may change without notice
- Results should not be used for operational or decision-making purposes
---

## Example Usage

The included example notebook DiscoCH_Demo.ipynb demonstrates the DiscoCH pipeline.

*IMPORTANT: The current swissEO version does not support full model implementation.
Sentinel-2 Band 7 is not currently available from swissEO and has been temporarily replaced for 
demonstration purposes only!
Noise in the raw SwissEO time series can also result in anomalous high and low discoloration probabilities.
Future SwissEO versions should support a more stable and accurate operational application. 
See [Bloom et al., in Review](http://dx.doi.org/10.2139/ssrn.5343751) for examples of accurate model 
application using the pre-processed Sentinel-2 time series of
[Koch et al., 2024](https://www.doi.org/10.16904/envidat.511)*

You can run the example notebook in Google Colab at the following link. No coding or local installation is required but 
you will need a free google account.
https://colab.research.google.com/github/ckbloom/DiscoCH/blob/master/notebooks/DiscoCH_Demo.ipynb

Note: Colab will likely throw a dependency error the first time you run the notebook.
Colab ships with preinstalled ML packages that require numpy>=2 but DiscoCH requires numpy<2.
Simply follow the prompts to restart the kernel and re-run the cells.

---

## Local Installation

If you just want to try out the tools, simply open the colab link above and run the cells.
If you want to run the code locally on your own machine, clone the repository using git in the command line
and create a new environment:
```bash
# Clone the directory
git clone https://github.com/ckb1oom/DiscoCH

# Navigate into the directory
cd DiscoCH

# Create a new conda environment
conda create -n DiscoCH python=3.11 -c conda-forge

# Activate the conda environment
conda activate DiscoCH

# Install the required packages
pip install -r requirements.txt

# To use the example notebook install jupyter lab (or equivalent)
pip install jupyterlab

# Install the DiscoCH package in editable mode
pip install -e .

# Open jupyter lab and be sure to update the directory in the file paths
jupyter lab
```

---

## License

This project is released under the MIT Licence.
See the 'LICENSE' file for details.