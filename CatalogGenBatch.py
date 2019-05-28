from __future__ import print_function
import numpy as np
from datetime import datetime
from AnnularModule import *
import os

#-----------------------TODO: INPUT PARAMETERS---------------------------

Mrange = [14.0,15.7]    #Cluster mass range in log scale. 
                        #   Default [14.0,15.7], i.e., 1e14 - 5e15 Msun
Crange = [4.0,4.0]     #Default [4.0,4.0], i.e. fixed at c=4.0 
Ngalaxies = 2000        #Default 4000
imgsize = 5000          #Default 5000 pixels
rmsNoise = 0.1          #Default 0.3, or 0 for a no-noise catalog.
NCatalog = 10000        #Default 10000
targetdir = "./NewCat/"

#-----------------------START GENERATING CATALOG-------------------------

center = [ imgsize/2, imgsize/2 ]

class Galaxy():
    def __init__(self):
        self.x = int(np.random.random()*imgsize)
        self.y = int(np.random.random()*imgsize)
        self.r = np.sqrt( (self.x - center[0])**2 + (self.y - center[1])**2 )
        self.ex = 0.0
        self.ey = 0.0

foldername = "Noise"+str(rmsNoise).replace('.','')+"_"+datetime.now().strftime("%Y%m%d%H%M")+"/" 
os.mkdir(targetdir+foldername)
print("Savinging catalogs on "+targetdir+foldername+" ...")
for ncat in range(NCatalog):
    if ncat % (NCatalog/20) == 0:
        print(" ", ncat, "/", NCatalog, end='\r')
    filename = targetdir+foldername+"Cat"+str(ncat)+".cat"
    f = open(filename,'w')

    # random sampling M200 from Mrange in log scale.
    M200 = 10.**( np.random.uniform(Mrange[0], Mrange[1]) )
    r200 = (M200/800/np.pi*3/rho_c)**(1/3.)     # in Mpc
    c = np.random.uniform(Crange[0], Crange[1])
    rs = r200/c/Dscale                          # in pixel
    
    CATALOG = [ Galaxy() for i in range(Ngalaxies) ]

    for i in CATALOG:
        r = i.r
        r2 = r*r
        dx = i.x - center[0]
        dy = i.y - center[1]
        cos2phi = (dx*dx - dy*dy)/r2
        sin2phi = 2.0*dx*dy/r2
        #ell = reduced_shear(r,rs,c)
        ell = shear_nfw(r,rs,c)
        i.ex = -ell*cos2phi
        i.ey = -ell*sin2phi
        if rmsNoise > 0:
            theta = 2*np.pi*np.random.random()
            theta2 = np.pi*np.random.random()
            Mag = np.sqrt(2)*rmsNoise*np.sin(theta2)
            i.ex += Mag*np.cos(2*theta)
            i.ey += Mag*np.sin(2*theta)
    

    f.write( '# fiat 1.0'
        +'\n# time created: '+str(datetime.now())
        +'\n# M200 = '+str(M200)
        +'\n# c = '+str(c)
        +'\n# zd = '+str(zd)
        +'\n# zs = '+str(zs)
        +'\n# center = '+str(center)
        +'\n# pixel_scale = '+str(pixel_scale)
        +'\n# ttype1 = coord_x'
        +'\n# ttype2 = coord_y'
        +'\n# ttype3 = ex'
        +'\n# ttype4 = ey' )

    for i in CATALOG:
        e = i.ex*i.ex + i.ey*i.ey
        if e < 2.0:
            f.write( '\n%-15f%-15f%-15f%-15f' %(i.x,i.y,i.ex,i.ey) ) 
    f.close()
print("\nDone")
#END
