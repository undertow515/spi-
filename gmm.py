# gmm.py
# This file contains functions for calculating the Standardized Precipitation Index (SPI) using Gaussian Mixture Models (GMM).
import numpy as np
from sklearn.mixture import GaussianMixture
import scipy.stats as stats
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import xarray as xr
import warnings
from d_utils import spi_cal_single_with_aic  # 이전에 개선된 함수 사용

def analyze_multimodality(data, station, window, feature_name, min_data_points=50):
    """
    각 관측소의 이동평균 데이터에 대한 다봉성 분석 (KDE 기반) - 개선된 버전
    
    Parameters:
    data (xarray.Dataset): 데이터셋
    station: 관측소 ID
    window (int): 윈도우 크기
    feature_name (str): 강수량 피처명
    min_data_points (int): 최소 데이터 포인트 수
    
    Returns:
    int: 피크 개수 (1, 2, 3)
    """
    
    # 해당 관측소의 이동평균 데이터 추출
    prcp_rolling = data[feature_name].sel(station=station).rolling(date=window, center=False).mean()
    rolling_data = prcp_rolling.dropna(dim='date').values
    kde = gaussian_kde(rolling_data)
    x_eval = np.linspace(rolling_data.min(), rolling_data.max(), len(rolling_data)*2)
    density = kde(x_eval)

    # KDE 결과에서 피크 찾기
    peaks, _ = find_peaks(density, distance=len(rolling_data) * 0.2)  # 또는 prominence=0.005 등 추가 조정 가능


    if len(peaks) <= 1:
        return 1
    elif len(peaks) >= 3:
        return 3
    else:
        return 2



def gmm_cdf(x, gmm, n_components):
    """
    GMM 모델의 CDF 계산 - 개선된 버전
    
    Parameters:
    x: 입력값들
    gmm: 학습된 GMM 모델
    n_components (int): 컴포넌트 개수
    
    Returns:
    array: CDF 값들
    """
    try:
        x = np.asarray(x)
        cdf = np.zeros_like(x, dtype=float)
        
        for i in range(n_components):
            weight = gmm.weights_[i]
            mean = gmm.means_[i][0]
            # 공분산이 0에 가까우면 최소값으로 보정
            variance = max(gmm.covariances_[i][0][0], 1e-8)
            std = np.sqrt(variance)
            
            # 각 컴포넌트의 CDF 기여도 계산
            component_cdf = stats.norm.cdf(x, mean, std)
            cdf += weight * component_cdf
        
        # CDF는 단조증가해야 하므로 보정
        cdf = np.clip(cdf, 0, 1)
        return cdf
        
    except Exception as e:
        print(f"GMM CDF calculation failed: {e}")
        return np.full_like(x, 0.5)  # 실패시 중간값 반환

def robust_gmm_fit(data, n_components, max_iter=200, n_init=5):
    """
    강건한 GMM 피팅
    """
    best_gmm = None
    best_bic = np.inf
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        for init in range(n_init):
            try:
                gmm = GaussianMixture(
                    n_components=n_components,
                    random_state=42 + init,
                    max_iter=max_iter,
                    reg_covar=1e-6,  # 수치적 안정성
                    init_params='kmeans'
                )
                
                gmm.fit(data.reshape(-1, 1))
                
                # 수렴 여부 확인
                if gmm.converged_:
                    bic = gmm.bic(data.reshape(-1, 1))
                    if bic < best_bic:
                        best_bic = bic
                        best_gmm = gmm
                        
            except Exception:
                continue
    
    return best_gmm

def calculate_gmm_aic(data, gmm, n_components):
    """
    GMM의 AIC 계산
    """
    try:
        return 2 * n_components * 2 - 2 * gmm.score(data.reshape(-1, 1)) * len(data)
    except:
        return np.inf

def cal_spi_gmm_single(data, station, window, feature_name):
    """
    단일 관측소에 대한 GMM 기반 SPI 계산 - 개선된 버전
    
    Parameters:
    data (xarray.Dataset): 데이터셋
    station: 관측소 ID  
    window (int): 윈도우 크기
    feature_name (str): 강수량 피처명
    
    Returns:
    tuple: (params dict, spi values, dates, model, aic)
    """
    try:
        # 해당 관측소의 이동평균 데이터 추출
        prcp_rolling = data[feature_name].sel(station=station).rolling(date=window, center=False).mean()
        valid_data = prcp_rolling.dropna(dim='date')
        
        if len(valid_data) == 0:
            return dict(), np.full_like(prcp_rolling, np.nan), prcp_rolling.date.values, None, np.inf
        
        rolling_values = valid_data.values
        dates = valid_data.date.values
        
        # 다봉성 분석
        n_components = analyze_multimodality(data, station, window, feature_name)
        
        # 단일 분포인 경우
        if n_components == 1:
            spi_values, aic_weibull = spi_cal_single_with_aic(
                station, stats.weibull_min, window, feature_name, data
            )
            return (dict(), 
                   spi_values, 
                   prcp_rolling.date.values, 
                   stats.weibull_min, 
                   aic_weibull)
        
        # 다봉 분포인 경우 GMM 적용
        else:
            gmm = robust_gmm_fit(rolling_values, n_components)

            
            # AIC 계산
            aic_gmm = calculate_gmm_aic(rolling_values, gmm, n_components)
            
            # GMM으로 SPI 계산

            cdf_values = gmm_cdf(rolling_values, gmm, n_components)
            
            # CDF 값 클리핑
            epsilon = 1e-6
            cdf_values = np.clip(cdf_values, epsilon, 1 - epsilon)

            # SPI 계산
            spi_values_valid = stats.norm.ppf(cdf_values)
            
            # 원본 데이터 크기에 맞춰 SPI 배열 생성
            full_spi = np.full(len(prcp_rolling), np.nan)
            valid_indices = ~np.isnan(prcp_rolling.values)
            full_spi[valid_indices] = spi_values_valid
            
            # 무한대 값 처리
            full_spi = np.where(np.isfinite(full_spi), full_spi, np.nan)
            
            params = {
                "weights": gmm.weights_.tolist(),
                "means": gmm.means_.flatten().tolist(),
                "stds": np.sqrt(gmm.covariances_).flatten().tolist(),
                "n_components": n_components,
                "converged": gmm.converged_
            }
            
            return params, full_spi, prcp_rolling.date.values, gmm, aic_gmm
            
    except Exception as e:
        print(f"GMM SPI calculation failed for station {station}: {e}")
        # 오류 시 기본 분포로 fallback
        try:
            spi_values, aic = spi_cal_single_with_aic(
                station, stats.weibull_min, window, feature_name, data
            )
            return dict(), spi_values, prcp_rolling.date.values, stats.weibull_min, aic
        except:
            return dict(), np.full_like(prcp_rolling, np.nan), prcp_rolling.date.values, None, np.inf


def analyze_multimodality_all_stations(data, window, feature_name):
    """
    모든 관측소의 다봉성 분석
    
    Parameters:
    data (xarray.Dataset): 데이터셋
    window (int): 윈도우 크기
    feature_name (str): 강수량 피처명
    
    Returns:
    dict: {station_id: peak_count}
    """
    multimodality_results = {}
    
    for station in data.station.values:
        try:
            peak_count = analyze_multimodality(data, station, window, feature_name)
            multimodality_results[station] = peak_count
            print(f"Station {station}: {peak_count} peaks detected")
        except Exception as e:
            print(f"Station {station}: 다봉성 분석 오류 - {str(e)}")
            multimodality_results[station] = 1  # 기본값
    
    return multimodality_results





def cal_spi_gmm_all_stations(data, window, feature_name):
    """
    모든 관측소에 대한 GMM 기반 SPI 계산 (AIC 포함)
    """
    # 이동평균 계산
    prcp_rolling = data[feature_name].rolling(date=window, center=False).mean()
    
    # 결과 저장용 배열 초기화
    spi_values = np.full_like(prcp_rolling.values, np.nan)
    aic_values = np.full(len(prcp_rolling.station), np.nan)
    params_dict = {}
    models_dict = {}
    
    # 각 관측소별로 처리
    for i, station in enumerate(prcp_rolling.station.values):
        try:
            # cal_spi_gmm_single은 (params, spi_values, dates, model, aic) 반환
            params, spi_station, dates, model, aic = cal_spi_gmm_single(
                data, station, window, feature_name
            )
            
            spi_values[i, :] = spi_station
            aic_values[i] = aic
            params_dict[station] = params
            models_dict[station] = model
            print(f"Station {station}: SPI 계산 완료, AIC={aic}")
            
        except Exception as e:
            print(f"Station {station}: 오류 발생 - {str(e)}")
            aic_values[i] = np.inf
            params_dict[station] = {}
            models_dict[station] = None
            continue
    
    # xarray DataArray 생성
    spi_data = xr.DataArray(
        spi_values,
        coords=prcp_rolling.coords,
        dims=prcp_rolling.dims,
        name='spi'
    )
    
    # AIC DataArray 생성
    aic_data = xr.DataArray(
        aic_values,
        coords=[prcp_rolling.station],
        dims=['station'],
        name='aic'
    )
    
    return spi_data, aic_data, params_dict, models_dict