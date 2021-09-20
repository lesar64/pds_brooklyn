# Yellowcab: A powerful data analysis toolkit for taxi rides in NYC

## What is it?

**yellowcab** is a Python package that provides a broad range of functions to work with 
the records of taxi rides in New York City. Specifically, it is aimed to work with a dataset 
of taxi rides obtained from [nyc.gov](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page). 
The project is mostly looking at the year 2020 with a strong focus on New York's most populous 
borough Brooklyn. Still, much of the functionality is easily reusable for other boroughs and years.

## Main Features
Here are just a few of the things that yellowcab does well:

  - Filtering the data in various ways to create a clean dataset, e.g. outliers or zero values
  - Adding various columns, e.g. regarding time, location or weather
  - Creating plots to visualize distributions and other statistics
  - Creating plots to further analyse the Covid-19 pandemic of 2020 and its heavy influence on mobility
  - Creating heatmaps, e.g. to visualize traffic in specific service zones
  - Offering functions for feature engineering, e.g. correlation analyses
  - Offering trained ML models to predict e.g. the distance of a trip, the amount of fares, the type 
    of payment and the upcoming needed capacity
   
## Documentation
The Documentation can be found in the accompanying [GitHub Wiki](https://github.com/lesar64/pds_brooklyn/wiki).

## Where to get it

Ensure that you have python 3.6 or higher installed. Furthermore, ensure that you 
have a package installer in the latest released version, e.g. 
[pip](https://pypi.org/project/pip/).

The source code is currently hosted on GitHub at:
https://github.com/lesar64/pds_brooklyn

Clone the repository and navigate to the folder \pds_brooklyn. Now execute:

```sh
# Execute
pip install -e .
# or
pip install yellowcab -e .
```
## Troubleshooting for dependeny installations

Sometimes, errors occur while automatically installing dependencies. 
The following tipps might help:


- Create a new environment by ```conda create --name myenv python=3.8``` with replacing ```myenv``` with 
a new name. You can check your python version with ```python -V```.
- Activate the environment by ```conda activate myenv``` with replacing ```myenv``` with 
the name.
- Ensure that the latest version of [pip](https://pypi.org/project/pip/) is installed.
  E.g. by executing ```conda install pip```
- [This Stackoverflow article](https://stackoverflow.com/questions/54149384/how-to-install-contextily) 
can help for problems with installing contextily. Download the **.whl** files for the **first 7 dependencies** (cf. chapter Dependencies). The files
can for example be obtained [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/). 
  Ensure that you obtain the files for the right python version. Move the files to the 
  folder  \pds_brooklyn.
Navigate to the folder in your Anaconda prompt and install the dependencies with pip
  in **exactly this order**  (starting with GDAL, ending with rasterio), each by 
```pip install filename``` with replacing ```filename``` by the name of the file. Execute 
  ```pip install contextily``` afterwards and again```pip install .``` (in the folder \pds_brooklyn)
- Pay attention to **trust a notebook**, before executing it (setting in the top right corner).
- Sometimes, package installation errors can be bypassed by instead executing 
  ```conda install -c conda-forge package``` to install a specific package.
- For Windows, [pipwin](https://pypi.org/project/pipwin/) seems to be able to solve issues. 
  - Especially 3 known problems with:
    - Fiona (GDAL API bug)
    - rasterio (GDAL API bug)
    - cartopy (cython version bug)
  - Run the following lines of code 
    in your Anaconda prompt if you have problems installing geopandas:
```sh
    pip install wheel
    pip install pipwin
    pipwin install numpy
    pipwin install pandas
    pipwin install shapely
    pipwin install gdal
    pipwin install fiona
    pipwin install pyproj
    pipwin install six
    pipwin install rtree
    pipwin install geopandas
    pipwin install rasterio
    pipwin install cython
    pipwin install cartopy
```


## Dependencies
- [NumPy](https://www.numpy.org)
- [pandas](https://pandas.pydata.org/)
- [Shapely](https://pypi.org/project/Shapely/)
- [GDAL](https://gdal.org/)
- [Fiona](https://pypi.org/project/Fiona/)
- [proj](https://proj.org/)
- [PyProj](https://pypi.org/project/pyproj/)
- [six](https://pypi.org/project/six/)
- [rtree](https://pypi.org/project/Rtree/)
- [geopandas](https://geopandas.org/)
- [cython](https://pypi.org/project/Cython/)
- [Cartopy](https://pypi.org/project/Cartopy/)
- [rasterio](https://rasterio.readthedocs.io/en/latest/)
- [contextily](https://contextily.readthedocs.io/en/latest/)
- [datetime](https://docs.python.org/3/library/datetime.html)
- [seaborn](https://seaborn.pydata.org/)
- [folium](https://pypi.org/project/folium/)
- [matplotlib](https://matplotlib.org/)
- [openpyxl](https://openpyxl.readthedocs.io/en/stable/)
- [pyarrow](https://pypi.org/project/pyarrow/)
- [setuptools](https://pypi.org/project/setuptools/)
- [scipy](https://pypi.org/project/scipy/)
- [click](https://click.palletsprojects.com/en/8.0.x/)
- [sklearn](https://scikit-learn.org/stable/)


## Background
Work on ``yellowcab`` started at the Faculty of Management, 
Economics, and Social Sciences at the University of Cologne. 
A team of five graduate students worked under supervision of 
Philipp Kienscherf within the scope of the module "Programming 
Data Science".

## Report
A report of the project with more detailed descriptions and its 
most interesting findings can be found [here](https://drive.google.com/file/d/1YRPSU653ais2UAjnsL81G2YjTOvK22l4/view?usp=sharing) and was also upload to Ilias by Steffen Weißhaar. 
