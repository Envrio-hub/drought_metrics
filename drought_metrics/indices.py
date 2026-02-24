import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import fisk, norm, pearson3
import matplotlib.pyplot as plt
from typing import Optional, Literal
from databases_companion.enum_variables import TemporalResolution


# Allowed distributions for SPEI fitting
SPEIDist = Literal["loglogistic", "pearson3"]


class DroughtIndices:
    """
    Drought index computation framework.

    Supports:
    - Aridity Index (AI)
    - Precipitation extremes analysis
    - Consecutive Dry Days (CDD)
    - SPI
    - SPEI (log-logistic or Pearson Type III)

    Input data may be hourly, daily, or monthly.
    """

    # ------------------------------------------------------------------
    # INITIALIZATION
    # ------------------------------------------------------------------
    def __init__(
        self,
        data_frame: pd.DataFrame,
        date_col: str = "date",
        date_format: str = "%Y-%m-%d",
        precip_col: str = "precip",
        eto_col: Optional[str] = None,
        temporal_analysis=TemporalResolution.daily,
    ):
        """
        Parameters
        ----------
        data_frame : pd.DataFrame
            Input time series dataset.
        date_col : str
            Name of datetime column.
        precip_col : str
            Column containing precipitation values.
        eto_col : str or None
            Column containing reference evapotranspiration.
        temporal_analysis : TemporalResolution
            Temporal resolution of input data.
        """

        # --- Step 1: Validate required columns ---
        if date_col not in data_frame.columns:
            raise KeyError(f"Missing required date column: {date_col}")

        if precip_col not in data_frame.columns:
            raise KeyError(f"Missing precipitation column: {precip_col}")

        # --- Step 2: Convert date column to datetime ---
        df = data_frame.copy()
        df[date_col] = pd.to_datetime(df[date_col], format=date_format)

        # --- Step 3: Sort chronologically and set datetime index ---
        df = df.sort_values(date_col).set_index(date_col)

        # --- Step 4: Store attributes ---
        self.df = df
        self.precip_col = precip_col
        self.eto_col = eto_col
        self.temporal_resolution = temporal_analysis

    # ------------------------------------------------------------------
    # INTERNAL RESAMPLING UTILITY
    # ------------------------------------------------------------------
    def _resampling_to_monthly(self):
        """
        Convert precipitation (and optionally PET) to monthly totals.

        Returns
        -------
        precip : pd.Series
            Monthly precipitation totals.
        eto : pd.Series or None
            Monthly evapotranspiration totals (if available).
        """

        precip = None
        eto = None

        # --- Step 1: Hourly data → daily → monthly ---
        if self.temporal_resolution == TemporalResolution.hourly:
            precip_daily = self.df[self.precip_col].resample("D").sum()
            precip = precip_daily.resample("ME").sum()

            if self.eto_col:
                eto_daily = self.df[self.eto_col].resample("D").sum()
                eto = eto_daily.resample("ME").sum()

        # --- Step 2: Daily data → monthly ---
        elif self.temporal_resolution == TemporalResolution.daily:
            precip = self.df[self.precip_col].resample("ME").sum()

            if self.eto_col:
                eto = self.df[self.eto_col].resample("ME").sum()

        # --- Step 3: Monthly data (already aggregated) ---
        elif self.temporal_resolution == TemporalResolution.monthly:
            precip = self.df[self.precip_col]

            if self.eto_col:
                eto = self.df[self.eto_col]

        else:
            return None, None

        return precip, eto

    # ------------------------------------------------------------------
    # ARIDITY INDEX
    # ------------------------------------------------------------------
    def aridity_index(self) -> pd.Series:
        """
        Compute annual Aridity Index (AI).

        AI = Annual ETo / Annual Precipitation
        """

        # --- Step 1: Ensure PET is available ---
        if not self.eto_col:
            raise ValueError("Reference evapotranspiration data are required.")

        # --- Step 2: Convert to annual totals ---
        if self.temporal_resolution != TemporalResolution.annual:
            annual_precip = self.df[self.precip_col].resample("YE").sum()
            annual_eto = self.df[self.eto_col].resample("YE").sum()
        else:
            annual_precip = self.df[self.precip_col]
            annual_eto = self.df[self.eto_col]

        # --- Step 3: Compute annual ratio ---
        ai = annual_eto / annual_precip

        return ai

    # ------------------------------------------------------------------
    # PRECIPITATION EXTREMES
    # ------------------------------------------------------------------
    def analyze_precip_extremes(self, lower_q=5.0, upper_q=95.0, make_plot=True):
        """
        Identify precipitation extremes using percentile thresholds.
        """

        # --- Step 1: Require daily resolution ---
        if self.temporal_resolution != TemporalResolution.daily:
            raise ValueError("Daily resolution required.")

        # --- Step 2: Remove missing values ---
        valid = self.df[self.precip_col].notna()
        precip_values = self.df.loc[valid, self.precip_col].values

        if precip_values.size == 0:
            raise ValueError("No valid precipitation data.")

        # --- Step 3: Compute percentile thresholds ---
        lower_thr = np.percentile(precip_values, lower_q)
        upper_thr = np.percentile(precip_values, upper_q)

        # --- Step 4: Classify events ---
        df_out = self.df.copy()
        df_out["extreme_class"] = "normal"
        df_out.loc[valid & (df_out[self.precip_col] <= lower_thr), "extreme_class"] = "low_extreme"
        df_out.loc[valid & (df_out[self.precip_col] >= upper_thr), "extreme_class"] = "high_extreme"

        # --- Step 5: Optional visualization ---
        if make_plot:
            sorted_vals = np.sort(precip_values)
            ecdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)

            plt.figure()
            plt.plot(sorted_vals, ecdf)
            plt.axvline(lower_thr, linestyle="--")
            plt.axvline(upper_thr, linestyle="--")
            plt.xlabel("Precipitation")
            plt.ylabel("Cumulative probability")
            plt.title("Empirical CDF")
            plt.grid(True)
            plt.show()

        return df_out, {"lower": lower_thr, "upper": upper_thr}

    # ------------------------------------------------------------------
    # CONSECUTIVE DRY DAYS
    # ------------------------------------------------------------------
    def extreme_drought(self, dry_threshold=1.0):
        """
        Compute consecutive dry day (CDD) statistics.
        """

        if self.temporal_resolution != TemporalResolution.daily:
            raise ValueError("Daily resolution required.")

        # --- Step 1: Identify dry days ---
        precip = self.df[self.precip_col]
        dry = precip.values <= dry_threshold

        # --- Step 2: Extract consecutive sequences ---
        cdd_lengths = []
        current_length = 0

        for is_dry in dry:
            if is_dry:
                current_length += 1
            else:
                if current_length > 0:
                    cdd_lengths.append(current_length)
                current_length = 0

        if current_length > 0:
            cdd_lengths.append(current_length)

        cdd_lengths = np.array(cdd_lengths)

        if len(cdd_lengths) == 0:
            raise ValueError("No dry spells detected.")

        # --- Step 3: Empirical CDF ---
        sorted_lengths = np.sort(cdd_lengths)
        cdf_vals = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)

        cdf = pd.DataFrame({"CDD_length": sorted_lengths, "CDF": cdf_vals})
        inv_cdf = pd.DataFrame({"CDD_length": sorted_lengths,
                                "ExceedanceProb": 1 - cdf_vals})

        return cdd_lengths, cdf, inv_cdf

    # ------------------------------------------------------------------
    # SPI
    # ------------------------------------------------------------------
    def compute_spi(self, scale=3, calib_start=None, calib_end=None):
        """
        Compute Standardized Precipitation Index (SPI).
        """

        if self.temporal_resolution == TemporalResolution.annual:
            raise ValueError("SPI requires sub-annual data.")

        # --- Step 1: Convert to monthly totals ---
        precip, _ = self._resampling_to_monthly()
        if precip is None:
            raise ValueError("Unsupported temporal resolution.")

        # --- Step 2: Rolling accumulation ---
        acc = precip.rolling(window=scale, min_periods=scale).sum()

        # --- Step 3: Select calibration period ---
        calib = acc.dropna()
        if calib_start and calib_end:
            calib = calib[calib_start:calib_end]

        if calib.empty:
            raise ValueError("Calibration period has no data.")

        # --- Step 4: Gamma fit (non-zero values) ---
        calib_nonzero = calib[calib > 0]
        q_zero = 1.0 - len(calib_nonzero) / len(calib)

        shape, loc, scale_param = stats.gamma.fit(calib_nonzero, floc=0.0)

        # --- Step 5: Transform to standard normal ---
        spi_values = []
        for x in acc:
            if np.isnan(x):
                spi_values.append(np.nan)
                continue

            if x <= 0:
                H = q_zero
            else:
                G = stats.gamma.cdf(x, shape, loc=loc, scale=scale_param)
                H = q_zero + (1 - q_zero) * G

            H = np.clip(H, 1e-6, 1 - 1e-6)
            spi_values.append(norm.ppf(H))

        return pd.Series(spi_values, index=acc.index, name=f"SPI-{scale}")

    # ------------------------------------------------------------------
    # SPEI
    # ------------------------------------------------------------------
    def compute_spei(self, scale=3, calibration_start=None,
                     calibration_end=None, dist: SPEIDist = "loglogistic",
                     eps=1e-12):
        """
        Compute SPEI using log-logistic or Pearson Type III distribution.
        """

        if not self.eto_col:
            raise ValueError("ETo column required for SPEI.")

        # --- Step 1: Monthly climatic water balance ---
        precip, eto = self._resampling_to_monthly()
        D = precip - eto

        # --- Step 2: Rolling accumulation ---
        accD = D.rolling(window=scale, min_periods=scale).sum()

        # --- Step 3: Calibration subset ---
        cal = accD.copy()
        if calibration_start:
            cal = cal[cal.index >= pd.to_datetime(calibration_start)]
        if calibration_end:
            cal = cal[cal.index <= pd.to_datetime(calibration_end)]

        spei = pd.Series(index=accD.index, dtype=float)

        # --- Step 4: Month-wise distribution fitting ---
        for m in range(1, 13):
            x_all = accD[accD.index.month == m].dropna()
            x_cal = cal[cal.index.month == m].dropna()

            if len(x_cal) < 20:
                spei.loc[x_all.index] = np.nan
                continue

            if dist == "loglogistic":
                c, loc, scale_ = fisk.fit(x_cal)
                p = fisk.cdf(x_all, c, loc=loc, scale=scale_)
            else:
                skew, loc, scale_ = pearson3.fit(x_cal)
                p = pearson3.cdf(x_all, skew, loc=loc, scale=scale_)

            p = np.clip(p, eps, 1 - eps)
            spei.loc[x_all.index] = norm.ppf(p)

        spei.name = f"SPEI-{scale}"
        return spei
    
    # ------------------------------------------------------------------
    # WASP (Weighted Anomaly Standardized Precipitation)
    # ------------------------------------------------------------------
    def compute_wasp(
        self,
        scale: int = 3,
        calibration_start: str | None = None,
        calibration_end: str | None = None,
    ) -> pd.Series:
        """
        Compute Weighted Anomaly Standardized Precipitation (WASP).

        Parameters
        ----------
        scale : int
            Accumulation window in months.
        calibration_start, calibration_end : str or None
            Optional calibration period (e.g. '1981-01', '2010-12').

        Returns
        -------
        wasp : pd.Series
            WASP time series.
        """

        # --- Step 1: Ensure suitable temporal resolution ---
        if self.temporal_resolution == TemporalResolution.annual:
            raise ValueError("WASP requires sub-annual (hourly, daily, or monthly) data.")

        # --- Step 2: Convert precipitation to monthly totals ---
        precip, _ = self._resampling_to_monthly()
        if precip is None:
            raise ValueError("Unsupported temporal resolution.")

        # --- Step 3: Define calibration subset ---
        calib = precip.copy()

        if calibration_start:
            calib = calib[calib.index >= pd.to_datetime(calibration_start)]
        if calibration_end:
            calib = calib[calib.index <= pd.to_datetime(calibration_end)]

        if calib.empty:
            raise ValueError("Calibration period contains no data.")

        # --- Step 4: Compute long-term monthly climatology ---
        # Mean precipitation for each calendar month
        monthly_climatology = calib.groupby(calib.index.month).mean()

        # --- Step 5: Compute monthly anomalies ---
        anomalies = precip.copy()

        for m in range(1, 13):
            month_mask = precip.index.month == m
            anomalies.loc[month_mask] = (
                precip.loc[month_mask] - monthly_climatology.loc[m]
            )

        # --- Step 6: Apply rolling accumulation ---
        # Accumulate anomalies over specified time scale
        acc_anom = anomalies.rolling(window=scale, min_periods=scale).sum()

        # --- Step 7: Standardize accumulated anomalies ---
        # Use calibration period for standard deviation
        acc_calib = acc_anom.copy()
        if calibration_start:
            acc_calib = acc_calib[acc_calib.index >= pd.to_datetime(calibration_start)]
        if calibration_end:
            acc_calib = acc_calib[acc_calib.index <= pd.to_datetime(calibration_end)]

        std = acc_calib.std()

        if std == 0 or np.isnan(std):
            raise ValueError("Standard deviation is zero; cannot standardize WASP.")

        wasp = acc_anom / std
        wasp.name = f"WASP-{scale}"

        return wasp