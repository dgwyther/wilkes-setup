import numpy as np
try: 
    from pyroms import hgrid
except:
    print('')

from pyroms.vgrid import s_coordinate
from pyroms.vgrid import s_coordinate_2
from pyroms.vgrid import s_coordinate_4
from pyroms.vgrid import z_r
from pyroms.vgrid import z_w

def calc_z(Vtransform,Vstretching,theta_s,theta_b,Tcline,hc,N,hh,zz):
    """
    function calc_z(Vtransform,Vstretching,theta_s,theta_b,Tcline,hc,N,hh,zz)
    
    usage:
    z_rho,z_w,=calc_z(Vtransform,Vstretching,theta_s,theta_b,Tcline,hc,N,h,zice)

    """
    if (Vtransform==2) & (Vstretching==4):
        grd_verts = s_coordinate_4(hh,theta_b,theta_s,Tcline,N)
        Cs_r_3 = (np.tile(grd_verts.Cs_r,[hh.shape[0],hh.shape[1],1]))
        S_rho_3 = (np.tile(grd_verts.s_rho,[hh.shape[0],hh.shape[1],1]))
        Cs_w_3 = (np.tile(grd_verts.Cs_w,[hh.shape[0],hh.shape[1],1]))
        S_w_3 = (np.tile(grd_verts.s_w,[hh.shape[0],hh.shape[1],1]))
        h_3 = (np.tile(hh,[N,1,1])).transpose(1,2,0)
        h_w3 = (np.tile(hh,[N+1,1,1])).transpose(1,2,0)
        zice_3 = (np.tile(zz,[N,1,1])).transpose(1,2,0)
        zice_w3 = (np.tile(zz,[N+1,1,1])).transpose(1,2,0)
    
        
        Zo_rho = (hc * S_rho_3 + Cs_r_3 * h_3) / (hc + h_3)
        z_rho = zice_3 + (zice_3 + h_3) * Zo_rho
        Zo_w = (hc * S_w_3 + Cs_w_3 * h_w3) / (hc + h_w3)
        z_w = Zo_w * (zice_w3 + h_w3) + zice_w3
        
        z_w = z_w.transpose(2,0,1)
        z_rho = z_rho.transpose(2,0,1)
        return(z_rho,z_w)


def rx0(h,rmask):
    """
    function rx0 = rx0(h,rmask)
 
    This function computes the bathymetry slope from a SCRUM NetCDF file.

    On Input:
       h           bathymetry at RHO-points.
       rmask       Land/Sea masking at RHO-points.
 
    On Output:
       rx0         Beckmann and Haidvogel grid stiffness ratios.
    """

    Mp, Lp = h.shape
    L=Lp-1
    M=Mp-1

    #  Land/Sea mask on U-points.
    umask = np.zeros((Mp,L))
    for j in range(Mp):
        for i in range(1,Lp):
            umask[j,i-1] = rmask[j,i] * rmask[j,i-1]

    #  Land/Sea mask on V-points.
    vmask = np.zeros((M,Lp))
    for j in range(1,Mp):
        for i in range(Lp):
            vmask[j-1,i] = rmask[j,i] * rmask[j-1,i]

    #-------------------------------------------------------------------
    #  Compute R-factor.
    #-------------------------------------------------------------------

    hx = np.zeros((Mp,L))
    hy = np.zeros((M,Lp))

    hx = abs(h[:,1:] - h[:,:-1]) / (h[:,1:] + h[:,:-1])
    hy = abs(h[1:,:] - h[:-1,:]) / (h[1:,:] + h[:-1,:])

    hx = hx * umask
    hy = hy * vmask

    rx0 = np.maximum(np.maximum(hx[:-1,:],hx[1:,:]),np.maximum(hy[:,:-1],hy[:,1:]))

    rmin = rx0.min()
    rmax = rx0.max()
    ravg = rx0.mean()
    rmed = np.median(rx0)

    print('  ')
    print('Minimum r-value = ', rmin)
    print('Maximum r-value = ', rmax)
    print('Mean    r-value = ', ravg)
    print('Median  r-value = ', rmed)

    return rx0



def rx1(z_w,rmask):
    """
    function rx1 = rx1(z_w,rmask)
 
    This function computes the bathymetry slope from a SCRUM NetCDF file.

    On Input:
       z_w         layer depth.
       rmask       Land/Sea masking at RHO-points.
 
    On Output:
       rx1         Haney stiffness ratios.
    """

    N, Lp, Mp = z_w.shape
    L=Lp-1
    M=Mp-1

    #  Land/Sea mask on U-points.
    umask = np.zeros((L,Mp))
    for j in range(Mp):
        for i in range(1,Lp):
            umask[i-1,j] = rmask[i,j] * rmask[i-1,j]

    #  Land/Sea mask on V-points.
    vmask = np.zeros((Lp,M))
    for j in range(1,Mp):
        for i in range(Lp):
            vmask[i,j-1] = rmask[i,j] * rmask[i,j-1]

    #-------------------------------------------------------------------
    #  Compute R-factor.
    #-------------------------------------------------------------------

    zx = np.zeros((N,L,Mp))
    zy = np.zeros((N,Lp,M))

    for k in range(N):
        zx[k,:] = abs((z_w[k,1:,:] - z_w[k,:-1,:] + z_w[k-1,1:,:] - z_w[k-1,:-1,:]) / 
                      (z_w[k,1:,:] + z_w[k,:-1,:] - z_w[k-1,1:,:] - z_w[k-1,:-1,:]))
        zy[k,:] = abs((z_w[k,:,1:] - z_w[k,:,:-1] + z_w[k-1,:,1:] - z_w[k-1,:,:-1]) /
                      (z_w[k,:,1:] + z_w[k,:,:-1] - z_w[k-1,:,1:] - z_w[k-1,:,:-1]))
        zx[k,:] = zx[k,:] * umask
        zy[k,:] = zy[k,:] * vmask


    r = np.maximum(np.maximum(zx[:,:,:-1],zx[:,:,1:]), np.maximum(zy[:,:-1,:],zy[:,1:,:]))

    rx1 = np.amax(r, axis=0)

    rmin = rx1.min()
    rmax = rx1.max()
    ravg = rx1.mean()
    rmed = np.median(rx1)

    print('  ')
    print('Minimum r-value = ', rmin)
    print('Maximum r-value = ', rmax)
    print('Mean    r-value = ', ravg)
    print('Median  r-value = ', rmed)

    return rx1