# DNN_Lensing
A deep neural network model to predict the galaxy cluster mass from gravitational lensing shear signal.
There are 2 sets of code for 2 data binning methods: circular binning and grid binning.
Circular binning is faster and more accurate, but requires circular symmetric galaxy cluster mass distribution.

### Instructions:
1. Get fiat catalogs and put them in the same directory. The catalog file should start with something like this:

```
# fiat 1.0
# created by Fri Jul 20 18:47:32 2018 by /usr/local/bin/fiatrecolumn 91-93,103-104,125-131,137-148 trial07_src.cat
# history: 
# ttype1 = id
# ttype2 = coord_ra
# ttype3 = coord_dec
# ttype4 = base_SdssCentroid_x
# ttype5 = base_SdssCentroid_y
# ttype6 = ext_shapeHSM_HsmPsfMoments_xx
# ttype7 = ext_shapeHSM_HsmPsfMoments_yy
# ttype8 = ext_shapeHSM_HsmPsfMoments_xy
# ttype9 = ext_shapeHSM_HsmShapeRegauss_e1
# ttype10 = ext_shapeHSM_HsmShapeRegauss_e2
```


2. Add a line specifying the cluster mass in each catalog, anywhere on the header of the catalog. You can simply output an additional line when you generate the catalog files, or edit the catalog if you didn't generate the catalog yourself. The format is:
```
# M200 = <cluster mass in Msun>
```


3. Run the CatToFeature.py (or CatToGridFeature.py in case of grid binning). All
catalog files must be in the same directory for this code to work. The output
is data.csv, located in the catalog directory. The run command is:
```
python CatToFeatures.py [optional arguments] <Path to catalog
directory>/<Filename of the first catalog> xcenter ycenter  
```
The optional arguments are the same as that of PythonAnnular.py. For example,
```
python CatToFeatures.py -c 'base_SdssCentroid_x base_SdssCentroid_y
ext_shapeHSM_HsmShapeRegauss_e1 ext_shapeHSM_HsmShapeRegauss_e2' -e 2500 -n 10
./CatDirectory/Cat0.cat 2500 2500
```
-c specify the column to use

-e specify the maximum radius in pixels

-n specify the number of bins


4. Run the NN code: Lensing_Dnn_Regression.ipynb (or Lensing_Dnn_Regression_Grid.ipynb in case of grid binning). You will need numpy, matplotlib, pandas, sklearn, and tensorflow, which are all available on google's colab. The input here is the csv file containing 
feature columns and target, which is the output of step 3.
