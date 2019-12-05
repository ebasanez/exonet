# - Requirements for vanilla installation:
# pip install lightkurve
#
# - Requirements for anaconda environemt
# conda install --channel conda-forge lightkurve
#
from math import floor
from math import ceil
import numpy as np
try:
    import lightkurve as lk
    from lightkurve import LightCurveCollection
    from lightkurve import search_lightcurvefile
except:
    import lightKurve as lk
    from lightKurve import LightCurveCollection
    from lightkurve import search_lightcurvefile

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
                         fold_mode = 'split', 
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
            lcc = LightCurveCollection([lc for lc in lcs if lc.mission == mission])
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
        phased_duration = duration / period
        lcs_local = [l[(l.phase > - local_view_size * phased_duration) & (l.phase < local_view_size * phased_duration)] for l in lcs_folded]
        
        lcs_bin_global = []
        lcs_bin_local = []
        
        print("Normalizing and binning global and local views")
        for i in range(len(lcs_folded)):
            try:
                # Bin global view
                fold_global = lcs_folded[i]
                lc_bin_global = fold_global.bin(binsize = fold_global.flux.size / global_bin_size, method = 'median')
                # Bin local view
                fold_local = lcs_local[i]
                lc_bin_local = fold_local.bin(binsize = fold_local.flux.size / local_bin_size, method = 'median')    
                
                # Normalize using median
                if normalize:
                    global_median = np.nanmedian(lc_bin_global.flux)
                    local_min = lc_bin_local.flux.min()
                    normalizeFunction = lambda f: (f - global_median) / (global_median - local_min)
                    lc_bin_global.flux = normalizeFunction(lc_bin_global.flux)
                    lc_bin_local.flux = normalizeFunction(lc_bin_local.flux)
                
                # Add folds to result
                lcs_bin_local.append(lc_bin_local)
                lcs_bin_global.append(lc_bin_global)
                
            except:
                print(f"DEATH&DESTRUCTION IN {i}! global lenght {len(lcs_folded[i])}, local length {len(lcs_local[i])}")
        
        # Create tensor with both global and local view
        result = []
        for i in range(len(lcs_bin_global)):
            result.append(np.append(lcs_bin_global[i].flux,lcs_bin_local[i].flux))
            
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
    
        
if __name__ == '__main__':
    import sys
    (script, koi_name, t0, period, duration) = sys.argv
    lkc = LightKurveClient()
    tensor = lkc.getKOILightKurve(koi_name, float(t0), float(period), float(duration), 2049+1e-9, 257+1e-9, 4, fold_mode = 'split', quarter = 1)
    print(tensor)

