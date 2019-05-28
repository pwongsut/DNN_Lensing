import numpy as np
import sys

#----------------------------Constants-----------------------------------
_G = 4.302e-9    #Gravitational constant in Mpc Msun^-1 (km/s)^2
_c0 = 299792.    #Speed of light in km/s.

#--------------------Cluster specific constants--------------------------
pixel_scale = 0.2   # in arcsecond/pixel
zd = 0.3            #Redshift at the cluster.
zs = 1.5            #Redshift at the background sources.
H = 79.56           #Hubble constant at the the cluster(z=zd), in km/s/Mpc. See http://home.fnal.gov/~gnedin/cc/
D_d = 926.9         #Angular size distance to the cluster in Mpc. See http://www.astro.ucla.edu/~wright/CosmoCalc.html
D_s = 1774.3        #Angular size distance to the background sources.
rmax = 2500.        #Outer radius of the outermost bin in pixels

D_ds = ((1+zs)*D_s - (1+zd)*D_d)/(1+zs)
rho_c = 3.0*H*H/(8*np.pi*_G)                  # in Msun Mpc^-3
Sigma_c = _c0*_c0/(4*np.pi*_G)*D_s/D_d/D_ds   # in Msun Mpc^-2
Dscale = pixel_scale*D_d/206265               # in Mpc/pixel

sys.stderr.write( '----------------------------------------------------------------\n')
sys.stderr.write( 'Import Annular module with the default parameters:\n')
sys.stderr.write( 'rho_c   = ' + str(rho_c) + ' Msun Mpc^-3\n')
sys.stderr.write( 'Sigma_c = ' + str(Sigma_c) + ' Msun Mpc^-2\n')
sys.stderr.write( 'Dscale  = ' + str(Dscale) + ' Mpc/pixel\n')
sys.stderr.write( 'rmax    = ' + str(rmax) + ' pixel\n')
sys.stderr.write( '\n*Dscale = pixel_scale(radian/pixel)*D_d(Mpc)\n')
sys.stderr.write( '----------------------------------------------------------------\n')

#If you need to change these parameters, do the followings:
#Import AnnularModule as Ann
#Ann.variable_name = new_value

#-------------------------Functions from NFW.py---------------------------
def delta_c(c):
    return 200.0/3*c*c*c/(np.log(1.+c)-c/(1.+c))

def rho_nfw(r,rs,c):    #NFW density profile, in Msun Mpc^-3
    x = r/rs
    return delta_c(c)*rho_c/x/(1+x)**2

def g(x):   # For calculating shear
    if x<1.0:
        return 8*np.arctanh(np.sqrt((1-x)/(1+x)))/(x*x*np.sqrt(1-x*x)) + 4/x/x*np.log(x/2) - 2/(x*x-1) + 4*np.arctanh(np.sqrt((1-x)/(1+x)))/(x*x-1)/np.sqrt(1-x*x)
    if x>1.0:
        return 8*np.arctan(np.sqrt((x-1)/(1+x)))/(x*x*np.sqrt(x*x-1)) + 4/x/x*np.log(x/2) - 2/(x*x-1) + 4*np.arctan(np.sqrt((x-1)/(1+x)))/(x*x-1)/np.sqrt(x*x-1)
    if x==1.0:
        return 10.0/3 + 4.0*np.log(0.5)

def g2(x):   # For calculating sigma
    if x<1.0:
        return 2.0*(1-2/np.sqrt(1-x*x)*np.arctanh(np.sqrt((1-x)/(1+x))))/(x*x-1)
    if x>1.0:
        return 2.0*(1-2/np.sqrt(x*x-1)*np.arctan(np.sqrt((x-1)/(1+x))))/(x*x-1)
    if x==1.0:
        return 2.0/3

def g3(x):   # For calculating average sigma inside x
    if x<1.0:
        return 8*np.arctanh(np.sqrt((1-x)/(1+x)))/(x*x*np.sqrt(1-x*x)) + 4/x/x*np.log(x/2)
    if x>1.0:
        return 8*np.arctan(np.sqrt((x-1)/(1+x)))/(x*x*np.sqrt(x*x-1)) + 4/x/x*np.log(x/2)
    if x==1.0:
        return 4.0 + 4*np.log(0.5)

def g3mod(x):   # Modified version for calculating mass from x'=0 to x, g3mod = g3*x*x 
    if x<1.0:
        return 8*np.arctanh(np.sqrt((1-x)/(1+x)))/(np.sqrt(1-x*x)) + 4*np.log(x/2)
    if x>1.0:
        return 8*np.arctan(np.sqrt((x-1)/(1+x)))/(np.sqrt(x*x-1)) + 4*np.log(x/2)
    if x==1.0:
        return 4.0 + 4*np.log(0.5)

def getM200(rs,c):
    r200 = c*rs*Dscale                  # in Mpc
    return 800.0*np.pi/3*rho_c*r200**3  # in Msun

def getrs(M200,c):     # rs in pixel
    return (M200/800.0/np.pi*3/rho_c)**(1/3.0) /c/Dscale


def Mbin(r1,r2,rs,c): # Calculate mass (in Msun) between r1 and r2
    if r1 == 0.0:
        return (rs*Dscale)**3*delta_c(c)*rho_c*(g3mod(r2/rs))*np.pi
    return (rs*Dscale)**3*delta_c(c)*rho_c*(g3mod(r2/rs)-g3mod(r1/rs))*np.pi

def prefactor(rs,c):  # rs in pixel
    return rs*delta_c(c)*rho_c/Sigma_c*Dscale


#-------------------------- Functions for Numpy array input ----------------------------------

def g_vec(x):
    y = np.zeros(x.shape)
    for i in range(len(y)):
        y[i] = g(x[i])
    return y

def g2_vec(x):
    y = np.zeros(x.shape)
    for i in range(len(y)):
        y[i] = g2(x[i])
    return y

def g3_vec(x):
    y = np.zeros(x.shape)
    for i in range(len(y)):
        y[i] = g3(x[i])
    return y

def g3mod_vec(x):
    y = np.zeros(x.shape)
    for i in range(len(y)):
        y[i] = g3mod(x[i])
    return y

def k12(r,rs,c): # Calculate ( average Sigma(x<x1) - average Sigma(x1<x<x2) ) / Sigma_c
    x1 = r/rs
    x2 = rmax/rs
    return prefactor(rs,c)*( g3_vec(x1) - (g3mod(x2)-g3mod_vec(x1))/(x2*x2-x1*x1) )

def k12_vec(r,rs,c): # Calculate ( average Sigma(x<x1) - average Sigma(x1<x<x2) ) / Sigma_c
    x1 = r/rs
    x2 = rmax/rs
    return prefactor(rs,c)*( g3_vec(x1) - (g3mod(x2)-g3mod_vec(x1))/(x2*x2-x1*x1) )

def convergence(r,rs,c):
    x = r/rs
    try:
        return rs*delta_c(c)*rho_c/Sigma_c*Dscale*g2(x)
    except ValueError:
        return rs*delta_c(c)*rho_c/Sigma_c*Dscale*g2_vec(x)

def shear_nfw(r,rs,c):          # r,rs in pixel
    x = r/rs
    try:
        return rs*delta_c(c)*rho_c/Sigma_c*Dscale*g(x)
    except ValueError:
        return rs*delta_c(c)*rho_c/Sigma_c*Dscale*g_vec(x)

def reduced_shear(r,rs,c):
    return shear_nfw(r,rs,c)/(1.-convergence(r,rs,c))

def sigma_sis(M200):            # input M200 in Msun, return sigma in km/s
    sigma2 = 2*np.pi*_G* (M200*M200*200.*rho_c/(48*np.pi*np.pi))**(1/3.)
    return np.sqrt(sigma2)

def shear_sis(r,sigma):         # r in pixel, sigma in km/s
    return 2*np.pi*sigma*sigma/(_c0*_c0)/(r*Dscale)*D_ds*D_d/D_s

