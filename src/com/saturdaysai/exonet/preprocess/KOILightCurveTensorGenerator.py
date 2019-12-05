
import sys
from pathlib import Path

import pandas as pd
import numpy as np

from math import floor
from math import ceil

import lightkurve as lk
from lightkurve import LightCurveCollection
from lightkurve import search_lightcurvefile


""" KOILightCurveTensorGenerator

This class extract Kepler Objects of Intesert from a feature dataset 
and uses LilghtKurveClient to extrart the light curves,
normalize those and store in fixed lenght tensors.
"""
class KOILightCurveTensorGenerator:
    
    DEFAULT_GLOBAL_LEN = 2049+1e-9 
    DEFAULT_LOCAL_LEN = 257+1e-9 
    DEFAULT_LOCAL_VIEW_WIDTH = 4
    DEFAULT_FOLD_MODE = 'fold'
    
    def __init__(self, 
                 source_file_name, 
                 destination_folder_path, 
                 global_tensor_len = DEFAULT_GLOBAL_LEN, 
                 local_tensor_len = DEFAULT_LOCAL_LEN, 
                 local_view_witdh = DEFAULT_LOCAL_VIEW_WIDTH, 
                 fold_mode = DEFAULT_FOLD_MODE):
        
        self.df = pd.read_csv(source_file_name)
        self.destination_folder_path = destination_folder_path
        self.global_tensor_len = int(global_tensor_len)
        self.local_tensor_len = int(local_tensor_len)
        self.local_view_witdh = int(local_view_witdh)
        self.fold_mode = fold_mode
        
    def getTensors(self, window = None):
        if window == None:  # If no window is given, retrieve the whole dataset
            df = self.df
        else:
            print(f"Window: {window}")
            df = self.df.loc[window[0]:window[1]]
        
        lkClient = LightKurveClient()
        list_tensors_x = []
        list_tensors_y = []
        list_tensors_z = []
        labeled = True if 'koi_is_planet' in df.columns else False
        
        for index, row in df.iterrows():
            print(row)
            koi_label = row.koi_is_planet if labeled else 'n/a'
            mission = row.mission
            koi_id = row.koi_id
            koi_name = row.koi_name
            koi_t0 = row.koi_time0bk
            duration = row.koi_duration
            period = row.koi_period
            
            list_koi_tensors_x = lkClient.getKOILightKurve(
                                     koi_id,
                                     koi_t0, 
                                     period, 
                                     duration, 
                                     self.global_tensor_len, 
                                     self.local_tensor_len, 
                                     self.local_view_witdh, 
                                     mission = mission, 
                                     fold_mode = self.fold_mode)
            
            print(f"Obtained {len(list_koi_tensors_x)} tensors.")
            list_koi_tensors_y = [koi_label] * len(list_koi_tensors_x)
            list_koi_tensors_z = [koi_name] * len(list_koi_tensors_x)
            list_tensors_x = list_tensors_x + list_koi_tensors_x
            list_tensors_y = list_tensors_y + list_koi_tensors_y
            list_tensors_z = list_tensors_z + list_koi_tensors_z
        
        tensors_x = np.stack(list_tensors_x)
        tensors_y = np.stack(list_tensors_y)
        tensors_z = np.stack(list_tensors_z)
        return tensors_x, tensors_y, tensors_z
    
    def persist(self, x, y, z, suffix):
        path = Path(self.destination_folder_path)
        np.save(path / f"X_{self.fold_mode}_{suffix}.npy", x)
        np.save(path / f"Y_{self.fold_mode}_{suffix}.npy", y)
        np.save(path / f"Z_{self.fold_mode}_{suffix}.npy", z)


class LightKurveClient():
    
    def getKOILightKurve(self, 
                         koi_kic, 
                         t0, 
                         period, 
                         duration, 
                         global_bin_size, 
                         local_bin_size, 
                         local_view_size, 
                         quarter = None, 
                         mission = 'Kepler', 
                         fold_mode = 'fold', 
                         normalize = True):
        """Given a KOI, initial time and period, returns the lightcurves.
        
        This function will download the Kepler data from the requested star,
        flatten, clean it, fold it or split it, bin it into the requested
        global and local view bin sizes, and normalize it to the [-1, 0] range.
        
        Attributes:
            koi_kic (): 
            t0 ():
            period ():
            duration ():
            global_bin_size ():
            local_bin_size ():
            quarter ():
            mission ():
            fold_mode ():
            normalize ():
        
        Returns:
            list: list of lightcurves retrieved, as numpy arrays. If in fold 
                mode, it will be a len 1 list; in split mode it will be a list
                with all the valid period lightcurves for the object
        
        """
    
        if quarter == None:
            print(f"Obtaining all {mission} light curves for KOI {koi_kic}.")
            lcs = search_lightcurvefile('KIC ' + str(koi_kic)).download_all(quality_bitmask='hardest').PDCSAP_FLUX
            lcc = LightCurveCollection([lc for lc in lcs if (lc.mission == mission) and (lc.targetid == koi_kic)])
            lc_raw = lcc.stitch()
        else:
            print(f"Obtaining {mission} light curve for KOI {koi_kic} (quarter {quarter})")
            lc_raw = search_lightcurvefile('KIC ' + str(koi_kic),quarter=quarter).download(quality_bitmask='hardest').PDCSAP_FLUX
        
        print("Cleaning and flattening light curve")
        lc_flat = lc_raw.flatten()        
        
        lc_clean = lc_flat.remove_outliers(sigma=20, sigma_upper=4) 
        
        print("Folding/splitting light curve")
        lcs_folded = []
        if fold_mode == 'fold':
            lcs_folded = self.fold(lc_clean, t0, period)
            
        if fold_mode == 'split':
            lcs_folded = self.split(lc_clean, t0, period)
        
        # Generate local view
        #phased_duration = duration / period
        #lcs_local = [l[(l.phase > - local_view_size * phased_duration/2.) & (l.phase < local_view_size * phased_duration/2.)] for l in lcs_folded]
        
        lcs_bin_global = []
        lcs_bin_local = []
        print("Normalizing and binning global and local views")
        for lc_folded in lcs_folded:
            
            # Bin global view
            lc_bin_global = self.bin_lc(lc_folded, global_bin_size, 
                                        width_to_dist_ratio=1.1, 
                                        limits=(-0.5, 0.5), 
                                        method = 'median')
            
            # Bin local view
            local_phased_duration = duration / period * local_view_size
            lc_bin_local  = self.bin_lc(lc_folded, local_bin_size, 
                                        width_to_dist_ratio=1.1, 
                                        limits=(-local_phased_duration/2. ,local_phased_duration/2.), 
                                        method = 'median')
            print(f"Local Binned nans: {np.isnan(lc_bin_local).sum()}")
            print(f"Global Binned nans: {np.isnan(lc_bin_global).sum()}")
            # Normalize using median
            if normalize:
                global_median = np.nanmedian(lc_bin_global)
                local_min = np.nanmin(lc_bin_local)
                normalizeFunction = lambda f: (f - global_median) / (global_median - local_min)
                lc_bin_global = normalizeFunction(lc_bin_global)
                lc_bin_local = normalizeFunction(lc_bin_local)
                
            # Add folds to result
            lcs_bin_local.append(lc_bin_local)
            lcs_bin_global.append(lc_bin_global)
            
        # Create tensor with both global and local view
        result = []
        for i in range(len(lcs_bin_global)):
            result.append(np.append(lcs_bin_global[i], lcs_bin_local[i]))
        return result
        
    def fold(self, lc_flat, t0, period):
        print(f"Folding light curve by period = {period} and start = {t0}")
        lc_fold =  lc_flat.fold(period, t0)
        return [lc_fold,]
    
    def split(self, lc_flat, t0, period):
        print(f"Spliting light curve by period = {period} and start = {t0}")
        t_max = lc_flat.time.max()
        t_period_init = t0 - period / 2
        t_period_end = t_period_init + period
        lc_period_folds = []
        print(f"{t_period_end}-{t_max}")
        while t_period_end < t_max:
            t_period_init += period
            t_period_end += period
            period_mask = ((lc_flat.time > t_period_init) & (lc_flat.time < t_period_end))
            lc_period_fold = lc_flat[period_mask].fold(period = period, t0 = (t_period_init + period / 2))
            if len(lc_period_fold.flux) > 0:
                lc_period_folds.append(lc_period_fold)
        print(f"Generated {len(lc_period_folds)} split folds")
        return lc_period_folds

    def bin_lc(self, lc, num_bins, 
               width_to_dist_ratio=1., limits=(-0.5, 0.5), method='median'):
         
        print(f"Lightcurve nans: {np.isnan(lc.flux).sum()} out of {lc.flux.shape}")
        print(f"Lightcurve phase in range: [{lc.phase.min()}, {lc.phase.max()}]")
        domain_width = float(limits[1] - limits[0])
        bin_dist = domain_width/num_bins
        bin_width = width_to_dist_ratio * bin_dist
        bin_centers = np.linspace(limits[0], limits[1], num_bins)
        bin_values = []
        
        if method=='median':
            agg = np.nanmedian
        elif method=='mean':
            agg = np.nanmean
        
        for bin_center in bin_centers:
            bin_lower = bin_center - bin_width / 2.
            bin_upper = bin_center + bin_width / 2.
            flux_bin = lc[(bin_lower < lc.phase) & (lc.phase < bin_upper)].flux
            if len(flux_bin) > 0:    
                bin_values.append(agg(flux_bin, overwrite_input=True))
            else:
                bin_values.append(np.nan)
        bin_values = np.array(bin_values)
        print(f"Nans in bins: {np.isnan(bin_values).sum()} out of {bin_values.shape}")
        return bin_values
          
 
# Standalone paginated light kurve download from features dataset
if __name__ == '__main__':
	(script, source_file_name, destination_folder_path, window_init, window_end, window_size) = sys.argv

	if window_init == 0 and window_end == -1:
		window = None
		window_suffix = "full"
	else:
		window = [window_init, window_end]
		window_suffix = f"{window_init:05d}_{window_end:05d}"

	fold_mode = "fold"

	cont = True
	current = get_init
	while cont: 
		window_init = current
		window_end  = min(current + window_size - 1, get_end)
		window_suffix = f"{window_init:05d}_{window_end:05d}"
    
		tensorGenerator = KOILightCurveTensorGenerator(source_file_name, 
                                               destination_folder_path, 
                                               fold_mode = fold_mode)

		window = [window_init, window_end]
		# Get the tensors
		(x, y, z) = tensorGenerator.getTensors(window)

		# Store the results
		tensorGenerator.persist(x, y, z, window_suffix)
    
		current = window_end + 1
    
		cont = window_end < get_end