import scipy.stats as stats
import pandas as pd
import numpy as np
import xarray as xr
from d_utils import load_config, spi_cal_single, spi_cal_all_stations, identify_drought_events,  spi_cal_single_with_aic, spi_cal_all_stations_with_aic
from gmm import cal_spi_gmm_single, cal_spi_gmm_all_stations, analyze_multimodality_all_stations



class Engine:
    def __init__(self, config_path='./config/config.yaml'):
        self.config = load_config(config_path)
        self.feature_name = self.config['feature_name']
        self.data_path = self.config['data_path']
        self.results_path = self.config['results_path']
        self.data = xr.open_dataset(self.data_path)
        self.window = int
        self.dist = None
    
    def get_spi(self, station=None):
        """
        Get SPI for the specified station or all stations.

        Args:
            station (int, optional): Station ID. If None, calculates for all stations.

        Returns:
            xr.DataArray: SPI values for the specified station or all stations.
        """
        if self.dist is None:
            raise ValueError("Distance distribution not set. Please set it before calling this method.")
        
        if station is not None:
            # Original single station calculation
            return spi_cal_single(station, self.dist, self.window, self.feature_name, self.data)
        else:
            # All stations calculation
            return spi_cal_all_stations(self.dist, self.window, self.feature_name, self.data)
        
    def get_spi_gmm(self, station=None):
        """
        Get GMM-based SPI for the specified station or all stations.

        Args:
            station (int, optional): Station ID. If None, calculates for all stations.
            n_components (int): Number of GMM components.

        Returns:
            tuple: (spi_data as xarray.DataArray, params_dict, gmm_models_dict)
        """
        if station is not None:
            # Original single station GMM SPI calculation
            return cal_spi_gmm_single(self.data, station, self.window, self.feature_name)
        else:
            # All stations GMM SPI calculation
            return cal_spi_gmm_all_stations(self.data, self.window, self.feature_name)
        
    def get_peaks_by_station(self):
        """
        Analyze multimodality for all stations.

        Returns:
            dict: {station_id: peak_count}
        """
        return analyze_multimodality_all_stations(self.data, self.window, self.feature_name)
    
    def get_spi_with_aic(self, station=None):
        """
        Get SPI and AIC for the specified station or all stations.
        """
        if self.dist is None:
            raise ValueError("Distance distribution not set.")
        
        if station is not None:
            spi_vals, aic_val = spi_cal_single_with_aic(
                station, self.dist, self.window, self.feature_name, self.data
            )
            return spi_vals, aic_val
        else:
            return spi_cal_all_stations_with_aic(
                self.dist, self.window, self.feature_name, self.data
            )
            
    def get_spi_gmm_with_aic(self, station=None):
        """
        Get GMM-based SPI and AIC for the specified station or all stations.
        """
        if station is not None:
            return cal_spi_gmm_single(self.data, station, self.window, self.feature_name)
        else:
            return cal_spi_gmm_all_stations(self.data, self.window, self.feature_name)
    