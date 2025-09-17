import yaml
import os
import numpy as np
import xarray as xr
from scipy import stats


def load_config(config_path='./config/config.yaml'):
    """
    Load configuration from a YAML file.
    
    Args:
        config_path (str): Path to the YAML configuration file.
        
    Returns:
        dict: Configuration parameters as a dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config



def spi_cal_single(station, dist, window, feature_name, data) -> xr.DataArray:
    """
    Calculate the Standardized Precipitation Index (SPI) for given coordinates and distance.
    
    Parameters:
    station (int): Station ID.
    dist (int): Distance in kilometers to consider for SPI calculation.
    window (int): Window size for SPI calculation.
    data (xarray.Dataset): Dataset containing precipitation data.
    
    Returns:
    xr.DataArray: SPI values for the specified coordinates.
    """
    prcp_rolling = data[feature_name].sel(station=station).rolling(date=window, center=False).mean()
    params = dist.fit(prcp_rolling.dropna(dim='date').values)
    # np.clip(cdf_vals, 1e-6, 1-1e-6)
    cdf_values = dist.cdf(prcp_rolling.dropna(dim='date').values, *params).clip(1e-6, 1-1e-6)
    val = stats.norm.ppf(cdf_values)

    mask = np.full_like(prcp_rolling, np.nan)

    # print(mask.shape, prcp_rolling.shape, val.shape)

    mask[prcp_rolling.notnull()] = val

    dates = prcp_rolling.date.values
    spi = xr.DataArray(mask, coords=[dates], dims=['date'], name='spi')

    return spi

def spi_cal_all_stations(dist, window, feature_name, data) -> xr.DataArray:
    """
    Calculate the Standardized Precipitation Index (SPI) for all stations at once.
    
    Parameters:
    dist: Distribution object (e.g., stats.gamma).
    window (int): Window size for SPI calculation.
    feature_name (str): Name of the precipitation feature in the dataset.
    data (xarray.Dataset): Dataset containing precipitation data.
    
    Returns:
    xr.DataArray: SPI values for all stations.
    """
    # Rolling mean for all stations
    prcp_rolling = data[feature_name].rolling(date=window, center=False).mean()
    
    # Initialize output array with NaN
    spi_values = np.full_like(prcp_rolling.values, np.nan)
    
    # Process each station
    for i, station in enumerate(prcp_rolling.station.values):
        station_data = prcp_rolling.sel(station=station).dropna(dim='date')
        
        if len(station_data) > 0:
            # Fit distribution parameters for this station
            params = dist.fit(station_data.values)
            
            # Calculate SPI values
            cdf_vals = dist.cdf(station_data.values, *params)
            # Avoid extreme values (0 and 1) that would cause infinite ppf values
            cdf_vals = np.clip(cdf_vals, 1e-6, 1-1e-6)
            spi_vals = stats.norm.ppf(cdf_vals)
            
            # Place SPI values back in the correct positions
            valid_mask = prcp_rolling.sel(station=station).notnull()
            spi_values[i, valid_mask] = spi_vals
    
    # Create xarray DataArray with same coordinates as input
    spi = xr.DataArray(
        spi_values, 
        coords=prcp_rolling.coords, 
        dims=prcp_rolling.dims, 
        name='spi'
    )
    
    return spi
import pandas as pd
def identify_drought_events(ds, window, station, threshold=-1.0):
    """
    identify_drought_events (Consecutive SPI < -1.0 during a specified window)
    
    Parameters:
    - ds: xr.DataArray
    - threshold: float, default -1.0 (SPI threshold for drought)
    - window: int, (Window size for SPI calculation)
    - station: int, (Station ID to analyze)
    
    Returns:
    - drought_events: DataFrame containing drought event information
    """
    
    # 특정 윈도우 크기의 데이터만 사용
    de = ds.sel(window=window, station=station).dropna(dim="date").to_dataframe().reset_index()
    de = de[["spi", "date"]]
    
    all_drought_events = []
    
        
    # 가뭄 이벤트 식별
    in_drought = False
    drought_start = None

    for idx, row in de.iterrows():
        if row['spi'] < threshold:
            if not in_drought:
                # 가뭄 시작
                in_drought = True
                drought_start = idx
        else:
            if in_drought:
                # 가뭄 종료
                in_drought = False
                
                # 가뭄 이벤트 정보 저장
                drought_period = de.iloc[drought_start:idx]
                duration = len(drought_period)  # 실제 지속 개월 수
                severity = -drought_period['spi'].sum()  # 누적 심도

                all_drought_events.append({
                    'station': station,
                    'window': window,
                    'start_date': drought_period.iloc[0]['date'],
                    'end_date': drought_period.iloc[-1]['date'],
                    'duration': duration,
                    'severity': severity,
                    'intensity': drought_period['spi'].mean(),
                    'min_spi': drought_period['spi'].min(),
                    'year': drought_period.iloc[0]['date'].year
                })
    return pd.DataFrame(all_drought_events)


import numpy as np
from scipy import stats
import warnings

def spi_cal_single_with_aic(station, dist, window, feature_name, data, min_positive=1e-6):
    """
    단일 분포를 사용한 SPI 계산 (AIC 포함) - 강건한 버전
    """
    prcp_rolling = data[feature_name].sel(station=station).rolling(date=window, center=False).mean()
    valid_data = prcp_rolling.dropna(dim='date').values
    
    if len(valid_data) == 0:
        return np.full_like(prcp_rolling, np.nan), np.inf
    
    try:
        # 1. 데이터 전처리
        clean_data = preprocess_data(valid_data, dist, min_positive)
        
        if len(clean_data) < 10:  # 최소 데이터 요구량
            return np.full_like(prcp_rolling, np.nan), np.inf
        
        # 2. 강건한 분포 피팅
        params, fitting_success = robust_distribution_fit(clean_data, dist)
        
        if not fitting_success:
            return np.full_like(prcp_rolling, np.nan), np.inf
        
        # 3. 안전한 AIC 계산
        aic = safe_aic_calculation(clean_data, dist, params)
        
        # 4. SPI 계산 (원본 데이터 사용)
        spi_values = calculate_spi(valid_data, dist, params)
        
        # 5. 원본 배열에 맞춰 결과 생성
        result = np.full_like(prcp_rolling, np.nan, dtype=float)
        valid_mask = prcp_rolling.notnull()
        result[valid_mask] = spi_values
        
        return result, aic
        
    except Exception as e:
        print(f"SPI calculation failed for station {station}: {e}")
        return np.full_like(prcp_rolling, np.nan), np.inf

def preprocess_data(data, dist, min_positive=1e-6):
    """
    분포별 데이터 전처리
    """
    clean_data = data[np.isfinite(data)]
    
    # Weibull, Gamma 등은 양수만 지원
    if hasattr(dist, 'a') and dist.a == 0:  # 양수 분포 체크
        clean_data = clean_data[clean_data > min_positive]
    
    # 극값 제거 (outlier 처리)
    if len(clean_data) > 20:  # 충분한 데이터가 있을 때만
        q1, q99 = np.percentile(clean_data, [1, 99])
        clean_data = clean_data[(clean_data >= q1) & (clean_data <= q99)]
    
    return clean_data

def robust_distribution_fit(data, dist):
    """
    강건한 분포 피팅
    """
    best_params = None
    best_loglik = -np.inf
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # 여러 방법으로 피팅 시도
        fitting_methods = []
        
        # 방법 1: 기본 MLE
        fitting_methods.append(lambda: dist.fit(data))
        
        # 방법 2: floc=0 고정 (Weibull, Gamma 등)
        if dist.name in ['weibull_min', 'gamma', 'lognorm']:
            fitting_methods.append(lambda: dist.fit(data, floc=0))
        
        # 방법 3: Method of Moments 초기값 (Weibull의 경우)
        if dist.name == 'weibull_min':
            def weibull_mom_fit():
                mean_data = np.mean(data)
                std_data = np.std(data)
                shape_init = max(0.5, (mean_data / std_data) ** 1.086)
                return dist.fit(data, f0=shape_init, floc=0)
            fitting_methods.append(weibull_mom_fit)
        
        for method in fitting_methods:
            try:
                params = method()
                
                # 피팅 품질 확인
                if is_valid_fit(data, dist, params):
                    loglik = np.sum(dist.logpdf(data, *params))
                    if np.isfinite(loglik) and loglik > best_loglik:
                        best_loglik = loglik
                        best_params = params
            except:
                continue
    
    return best_params, best_params is not None

def is_valid_fit(data, dist, params):
    """
    피팅 결과 유효성 검사
    """
    try:
        # 파라미터 유효성 검사
        if any(not np.isfinite(p) for p in params):
            return False
        
        # PDF 계산 가능성 확인
        pdf_values = dist.pdf(data, *params)
        if not np.all(np.isfinite(pdf_values)) or np.any(pdf_values <= 0):
            return False
        
        # CDF 계산 가능성 확인
        cdf_values = dist.cdf(data, *params)
        if not np.all(np.isfinite(cdf_values)):
            return False
        
        return True
    except:
        return False

def safe_aic_calculation(data, dist, params):
    """
    안전한 AIC 계산
    """
    try:
        lpdf = dist.logpdf(data, *params)
        
        # 무한대 또는 NaN 값 처리
        if not np.all(np.isfinite(lpdf)):
            finite_mask = np.isfinite(lpdf)
            if np.sum(finite_mask) == 0:
                return np.inf
            
            # 무한대 값을 최소 유한값으로 대체
            min_finite = np.min(lpdf[finite_mask])
            penalty = 50  # 큰 페널티
            lpdf = np.where(finite_mask, lpdf, min_finite - penalty)
        
        log_likelihood = np.sum(lpdf)
        k = len(params)
        aic = 2 * k - 2 * log_likelihood
        
        return aic if np.isfinite(aic) else np.inf
        
    except Exception as e:
        return np.inf

def calculate_spi(data, dist, params):
    """
    SPI 값 계산
    """
    try:
        # CDF 계산 및 클리핑
        cdf_values = dist.cdf(data, *params)
        
        # 극값 처리 (0과 1을 피함)
        epsilon = 1e-8
        cdf_values = np.clip(cdf_values, epsilon, 1 - epsilon)
        
        # 표준정규분포의 역CDF로 SPI 계산
        spi_values = stats.norm.ppf(cdf_values)
        
        # 무한대 값 제거
        spi_values = np.where(np.isfinite(spi_values), spi_values, np.nan)
        
        return spi_values
        
    except Exception as e:
        return np.full_like(data, np.nan)

# 사용 예시
"""
# 단일 스테이션 계산
spi_result, aic_value = spi_cal_single_with_aic(
    station=170, 
    dist=stats.weibull_min, 
    window=3, 
    feature_name='prcp', 
    data=data
)

print(f"AIC: {aic_value}")
print(f"SPI range: {np.nanmin(spi_result)} to {np.nanmax(spi_result)}")
"""


def spi_cal_all_stations_with_aic(dist, window, feature_name, data):
    """
    모든 관측소에 대한 SPI 계산 (AIC 포함)
    """
    # Rolling mean for all stations
    prcp_rolling = data[feature_name].rolling(date=window, center=False).mean()
    
    # Initialize output arrays
    spi_values = np.full_like(prcp_rolling.values, np.nan)
    aic_values = np.full(len(prcp_rolling.station), np.nan)
    
    # Process each station
    for i, station in enumerate(prcp_rolling.station.values):
        spi_station, aic_station = spi_cal_single_with_aic(
            station, dist, window, feature_name, data
        )
        spi_values[i, :] = spi_station
        aic_values[i] = aic_station
    
    # Create xarray DataArrays
    spi = xr.DataArray(
        spi_values, 
        coords=prcp_rolling.coords, 
        dims=prcp_rolling.dims, 
        name='spi'
    )
    
    aic = xr.DataArray(
        aic_values,
        coords=[prcp_rolling.station],
        dims=['station'],
        name='aic'
    )
    
    return spi, aic