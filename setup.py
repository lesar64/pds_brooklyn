from setuptools import setup

setup(
    name='yellowcab',
    version='1.0',
    description="Semester Project - Programming Data Science",
    author="Julian Fenten, Jan Fiedler, Pürlü Selman, Christian Schwendner, Steffen Weißhaar",
    author_email="student@uni-koeln.de",
    packages=["yellowcab"],
    install_requires=[
        'wheel',
        'pipwin',
        'NumPy',
        'pandas',
        'Shapely',
        'GDAL',
        'Fiona',
        'proj>=0.2.0',
        'pyproj',
        'six',
        'rtree',
        'geopandas',
        'cython>=0.15.1+',
        'Cartopy',
        'rasterio',
        'contextily',
        'datetime',
        'seaborn',
        'folium',
        'matplotlib',
        'openpyxl',
        'pyarrow',
        'setuptools',
        'scipy',
        'click',
        'sklearn',
        'torch',
        'pytorch_lightning'
    ],
    entry_points={
        'console_scripts': ['yellowcab=yellowcab.cli:main']
    }
)