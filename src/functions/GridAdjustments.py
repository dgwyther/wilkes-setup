def adjustWCT(h_new,zice_new,h_alter,zice_alter,eta,xi):
    """
    function adjustWCT(h,zice,h_alter,zice_alter,eta,xi
    
    usage:
    h_new, zice_new=adjustWCT(h,zice,h_alter,zice_alter,eta,xi)
    
    This function adjust h[eta,xi] + h_alter and zice[eta,xi]+zice_alter. The intended use would be to 
    thicken the water column at a set location. Given rx1 is a function of steepness, depth and wct,
    the intended use would be, for example:
    h_alter=200 #lower the bathymetry by 200 m
    zice_alter=-150 #lower (neg zice is lower) zice by only 150m
    So resultant water column here starts 200m lower and is 50m thicker than previously. 
    
    To pass a range, use the slice literal:
    eta=slice(100,132)

    """
    h_new[eta,xi] = h_new[eta,xi] + h_alter
    zice_new[eta,xi] = zice_new[eta,xi] + zice_alter
    return h_new, zice_new

def adjustMask(mask_new,mask_alter,eta,xi):
    """
    function adjustMask(mask_new,mask_alter,eta,xi)
    
    usage:
    mask_new=adjustMask(mask_new,mask_alter,eta,xi)
    
    This function adjusts mask at[eta,xi] to the new value mask_alter.
    e.g.
    mask[eta,xi] = mask_new,
    will change the value of a mask at [eta,xi] to the new value 
    mask_alter.

    """    
    mask_new[eta,xi]=mask_alter
    return mask_new



# in_val = zzz
# ocean_mask = mmm
# rx0val = 0.03
# Area = Area
# max_iterations = 20
# roi_row_min=395
# roi_row_max=423
# roi_col_min=240
# roi_col_max=280
# if_plotting  = 0
def smoothRegion(in_val,ocean_mask,rx0val,Area,max_iterations,roi_row_min,roi_row_max,roi_col_min,roi_col_max,if_plotting):
    import numpy as np 
    import matplotlib.pyplot as plt
    from ext.tools.smoothing_PlusMinus_rx0 import smoothing_PlusMinus_rx0

    msk = np.zeros(in_val.shape)

    msk[roi_row_min:roi_row_max,roi_col_min:roi_col_max] = 1
    msk = msk*ocean_mask

    if if_plotting:
        plt.pcolormesh(msk)
        plt.show()
    
    out_smooth, HmodifVal, ValueFct = smoothing_PlusMinus_rx0(msk,in_val,rx0val,Area,max_iterations)

    if if_plotting:
        plt.pcolormesh(out_smooth-in_val)
        plt.colorbar()
        plt.axis((roi_col_min-10,roi_col_max+10,roi_row_min-10,roi_row_max+10))
        plt.show()

        plt.pcolormesh(out_smooth)
        plt.colorbar()
        plt.axis((roi_col_min-10,roi_col_max+10,roi_row_min-10,roi_row_max+10))
        plt.show()

        # plt.pcolormesh((hhh+zzz)*msk)
        # plt.colorbar()
        # plt.axis((roi_col_min-10,roi_col_max+10,roi_row_min-10,roi_row_max+10))
        # plt.show()

        # plt.pcolormesh((hhh+zzz - out_smooth)*msk)
        # plt.colorbar()
        # plt.axis((roi_col_min-10,roi_col_max+10,roi_row_min-10,roi_row_max+10))
        # plt.show()

# hhh = hhh + (zzz - out_smooth)
# zzz = out_smooth
    return out_smooth




# in_z = zzz.copy()
# in_h = hhh.copy()
# in_msko = mmm.copy()
# roi_row_min=395
# roi_row_max=423
# roi_col_min=240
# roi_col_max=280
# if_plotting  = 0
# min_threshold_h = 100

def minWCTRegion(in_z,in_h,in_msko,roi_row_min,roi_row_max,roi_col_min,roi_col_max,if_plotting,min_threshold_h):
    import numpy as np 
    import matplotlib.pyplot as plt
        
    msk = np.zeros(in_h.shape)

    msk[roi_row_min:roi_row_max,roi_col_min:roi_col_max] = 1
    msk = (msk*in_msko).copy()

    if if_plotting:
        plt.pcolormesh(msk)
        plt.show()

    wct_thr = (in_z+in_h) < min_threshold_h

    msk_full = (wct_thr*msk)
    msk_full = msk_full==1


    out = in_h.copy()
    out[msk_full] = -in_z[msk_full] + min_threshold_h

    if if_plotting:
        plt.pcolormesh(msk_full)
        plt.colorbar()
        plt.axis((roi_col_min-10,roi_col_max+10,roi_row_min-10,roi_row_max+10))
        plt.title('msk_full')
        plt.show()

        plt.pcolormesh(out)
        plt.colorbar()
        plt.axis((roi_col_min-10,roi_col_max+10,roi_row_min-10,roi_row_max+10))
        plt.title('new h')
        plt.show()

        plt.pcolormesh((in_h+in_z)*msk)
        plt.colorbar()
        plt.axis((roi_col_min-10,roi_col_max+10,roi_row_min-10,roi_row_max+10))
        plt.title('old wct')
        plt.show()

        plt.pcolormesh((out - in_h))
        plt.colorbar()
        plt.axis((roi_col_min-10,roi_col_max+10,roi_row_min-10,roi_row_max+10))
        plt.title('diff in wct (new-old)')
        plt.show()

    in_h[roi_row_min:roi_row_max,roi_col_min:roi_col_max] = out[roi_row_min:roi_row_max,roi_col_min:roi_col_max]
    
    return in_h