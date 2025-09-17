from engine import Engine
from scipy import stats
import xarray as xr
from os import path
import pickle

# dist_dict = {
#     "gamma": stats.gamma,
#     "lognorm": stats.lognorm,
#     "weibull_min": stats.weibull_min,
#     "pearson3": stats.pearson3,
#     "norm" : stats.norm,
#     "gev" : stats.genextreme,
#     "gumbel": stats.gumbel_r,
# }

windows = [i*3 for i in range(1, 9)]  # 3, 6, 9, ..., 24 months

def main():
    # Initialize the engine with the configuration file
    engine = Engine(config_path='./config/config.yaml')
    engine.dist = stats.weibull_min

    # Calculate SPI for all stations with different window sizes
    l = []
    for window in windows:
        engine.window = window
        # Example: Get SPI for a specific station
        spi = engine.get_spi()
        # Merge SPI results into a single Dataset (add window dimension)
        spi["window"] = window
        l.append(spi)
    ds = xr.concat(l, dim="window")
    # Save the results to a NetCDF file
    ds.to_netcdf(path.join(engine.results_path, "spi_results_weibull.nc"))
    print("SPI results saved to:", path.join(engine.results_path, "spi_results_weibull.nc"))
    ds.close()

if __name__ == "__main__":
    main()


    
    