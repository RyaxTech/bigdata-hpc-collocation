#!/usr/bin/env python3
"""Prepare sentinel data for a given ROI between dates.

Usage:
    senprep.py create [-c]
    senprep.py list             --config=CONFIG_FILE [options]
    senprep.py download         --config=CONFIG_FILE [options]
    senprep.py process          --config=CONFIG_FILE [options]
    senprep.py download_process --config=CONFIG_FILE [options]

Commands:
    create        Create or clone an existing configuration file
    list          List SENTINEL products that match a configuration
    download      Download SENTINEL products that match a configuration
    process       Run processing on already-downloaded products
    pipeline      Run the full download+processing pipeline

Options:
    CREATE command (configurations)
    -c                               Let user paste geojson, rather than ask for filename

    LIST, DOWNLOAD, PROCESS, DOWNLOAD_PROCESS commands (satellite pipeline)
    --config CONFIG_FILE             File containing region file to preprocess
    --credentials CREDENTIAL_FILE    File containing sentinel API credentials [default: credentials.json]
    --rebuild                        Rebuild products
    --full_collocation               Whether collocate the whole product or only the roi
    --skip_week                      Skip all weeks that do not yield products covering complete ROI
    --primary primary_PRODUCT        Select primary product S1 or S2 [default: S2]
    --skip_secondary                 Skip the listing and processing of secondary product
    --external_bucket                Will check LTA products from AWS, Google or Sentinelhub
    --available_area                 Will list part of an ROI that matches the required specifications
"""
from datetime import date, timedelta
from functools import partial
from pathlib import Path

import json
import math
import subprocess
import os
import tempfile
import time
import logging

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import pyproj
import rasterio as rio
from IPython import display
from descartes.patch import PolygonPatch
from osgeo import gdal
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt, sentinel
from shapely.geometry import Polygon, MultiPolygon, shape
from shapely.ops import transform
import numpy as np

from google.cloud import storage
import urllib3
import shutil
import requests
# modules local to this src dir
import getpass

try:
    import configutil, roiutil, sen_plot
except:
    from src import configutil, roiutil, sen_plot


SENTINEL_ROOT = "/var/satellite-data/"
SENTINEL_STORAGE_PATH = "/var/satellite-data/Sentinel_Patches/"
DEBUG = False


def download_from_googlecloud(client, bucket, blob_prefix, productname, rootdir=SENTINEL_ROOT):
    """ Replacement for gsutil to recursively download from a bucket using the Google Storage Python API"""
    
    if not os.path.exists(SENTINEL_ROOT + productname):
        os.makedirs(SENTINEL_ROOT + productname)
    blobs =client.list_blobs(bucket, prefix = blob_prefix, delimiter='/')
    for blob in blobs:
        if "folder" in blob.name:
            prefix_new = blob.name[:-9]+'/'
            productname_new = productname + '/' + blob.name[:-9].split("/")[-1] 
            download_from_googlecloud(client, bucket, prefix_new, productname_new, SENTINEL_ROOT) 
        else:
            filename = blob.name.split("/")[-1]
            blob.download_to_filename(SENTINEL_ROOT + productname + '/' + filename)


def nearest_previous_monday(ddate):
    """Get the Monday before or on a given date.

    Weekday is from 0 (monday) to 6 (sunday)
    so subtract weekday from given date to find the nearest earlier Monday.

    If the passed date IS a Monday, this will be a noop
    i.e. it'll return the same date.
    """
    return ddate - timedelta(days=ddate.weekday())


def nearest_next_sunday(ddate):
    """Get the Sunday after or on a given date.

    Weekday is from 0 (monday) to 6 (sunday)
    so subtract weekday from given date to find the nearest earlier Monday.

    If the passed date IS a Sunday, this will be a noop
    i.e. it'll return the same date.
    """
    if ddate.weekday() == 6:
        return ddate
    return ddate - timedelta(days=ddate.weekday()) + timedelta(days=6)


def yyyymmdd_to_date(d):
    yyyy = int(d[:4])
    mm = int(d[4:6])
    dd = int(d[6:8])
    return date(yyyy, mm, dd)


def yyyymmdd_to_date(datestr):
    """Dumb conversion of a yyyymmdd string to date object."""
    year_4d = int(datestr[:4])
    month_2d = int(datestr[4:6])
    day_2d = int(datestr[6:8])
    return date(year_4d, month_2d, day_2d)


def load_api(credentials_json_file_path):
    """Load SentinelAPI with a users credentials."""
    credentials = json.load(open(credentials_json_file_path))
    return SentinelAPI(
        credentials["username"],
        credentials["password"],
        credentials["sentinel_url_endpoint"],
    )


def load_ROI(ROI_path):
    """
    Loads ROI in a shapely geometry
    Parameters:
    ROI_path: Path to the geojson file
    Returns shapely geometry
    """
    with open(ROI_path) as f:
        Features = json.load(f)["features"]

    # IF ROI is a collection of features
    # ROI =(GeometryCollection([shape(feature["geometry"]) for feature in scotland_features]))

    # IF ROI has a single feature
    for feature in Features:
        ROI = shape(feature["geometry"])

    return ROI


def find_S1(ROI_footprint, start_date, end_date, api, **kwargs):
    """
    Finds S1 products given Region of Interest and corresponding Sentinel2 product
    geopanda dataframe

    Parameters
    ----------
    s2_df: geopandas.DataFrame
         Returned from Sentinelsat api (potentially sorted)
    ROI: shapely geometry
        Region of Interest

    Keyword Arguments
    -----------------
    plot_tiles : bool [default: True]
        Whether to plot S1 product tiles

    Returns
    -------
    geopandas.DataFrame of s1 products
    """
    s1_products = api.query(
        ROI_footprint,
        date=(start_date, end_date),
        platformname="Sentinel-1",
        producttype="GRD",
    )

    s1_products_df = api.to_geodataframe(s1_products)
    if kwargs.get("plot_tiles", True):
        sen_plot.plot_Stiles_plus_ROI(ROI_footprint, s1_products_df, "blue", grid=False)
    n_prod = s1_products_df.shape[0]
    print(
        f"\nMatching Sentinel-1 Products found from {start_date} to {end_date}: {n_prod}"
    )
    return s1_products_df


def find_S2(ROI_footprint, start_date, end_date, api, **kwargs):
    """Finds S2 products given a Region of Interest.

    Arguments
    ---------
    ROI : shapely.shape
        Region of Interest as shapely geometry
    start_date: str
        format yyyymmdd e.g. 20200601 for 1st June 2020
    end_date: str
        format yyyymmdd e.g. 20200601 for 1st June 2020
    api : sentinelsat api

    Keyword-Arguments
    -----------------
    cloud_cover : tuple (int int)
        Lower and upper bound of acceptable cloud cover

    Returns
    -------
    s2_products_df : geopandas.DataFrame
    """
    cloud_cover = tuple(kwargs.get("cloud_cover", (0, 20)))
    s2_products = api.query(
        ROI_footprint,
        date=(start_date, end_date),
        platformname="Sentinel-2",
        cloudcoverpercentage=cloud_cover,
        producttype="S2MSI2A",
    )

    s2_products_df = api.to_geodataframe(s2_products)
    n_prod = s2_products_df.shape[0]

    print(
        f"\nMatching Sentinel-2 Products found from {start_date} to {end_date}: {n_prod}"
    )
    return s2_products_df


def find_S1_IW(ROI_footprint, start_date, end_date, api, **kwargs):
    """
    Finds SLC IW S1 products given Region of Interest and timeframe
    geopanda dataframe
    Parameters
    ----------
    ROI: shapely geometry
        Region of Interest
    start_date, end_date: date format
    api: sentinelsat api
    Keyword Arguments
    -----------------
    plot_tiles : bool [default: True]
        Whether to plot S1 product tiles
    Returns
    -------
    geopandas.DataFrame of s1 products
    """
    s1_products = api.query(
        ROI_footprint,
        date=(start_date, end_date),
        platformname="Sentinel-1",
        sensoroperationalmode="IW",
    )

    s1_products_df = api.to_geodataframe(s1_products)
    if kwargs.get("plot_tiles", True):
        sen_plot.plot_Stiles_plus_ROI(ROI_footprint, s1_products_df, "blue", grid=False)
    n_prod = s1_products_df.shape[0]
    print(
        f"\nMatching Sentinel-1 Products found from {start_date} to {end_date}: {n_prod}"
    )

    return s1_products_df


def find_S1_IW_old(s1_products_df, ROI_footprint, api, **kwargs):
    """
    Finds old S1 product given ONE current SAR IW product dataframe and Region of Interest
    Parameters
    ----------
    s1_products_df: geopandas.DataFrame
         Returned from Sentinelsat api
    ROI: shapely geometry
        Region of Interest
    api: sentinel api
    Keyword Arguments
    -----------------
    plot_tiles : bool [default: True]
        Whether to plot S1 product tiles
    Returns
    -------
    geopandas.DataFrame of one s1 product
    """

    s1_old_date = s1_products_df.beginposition.to_pydatetime()

    start_old_s1 = date(
        s1_old_date.year, s1_old_date.month, s1_old_date.day
    ) - timedelta(12)
    end_old_s1 = start_old_s1 + timedelta(1)
    slicenumber = int(s1_products_df.slicenumber)
    s1_old_products = api.query(
        date=(start_old_s1, end_old_s1),
        platformname=s1_products_df.platformname,
        relativeorbitnumber=s1_products_df.relativeorbitnumber,
        sensoroperationalmode=s1_products_df.sensoroperationalmode,
        producttype=s1_products_df.producttype,
        slicenumber=slicenumber,
    )

    s1_products_df = api.to_geodataframe(s1_old_products)
    if kwargs.get("plot_tiles", True):
        sen_plot.plot_Stiles_plus_ROI(ROI_footprint, s1_products_df, "blue", grid=False)
    n_prod = s1_products_df.shape[0]
    print(
        f"\nMatching Sentinel-1 Products found from {start_date} to {end_date}: {n_prod}"
    )

    return s1_products_df


def sort_sentinel_products(products, ROI, sorting_params, sorting_params_ascending):
    """
    Sort Sentinel tiles based on the common ROI geometry
    Returns panda dataframe with first column as product id and second column as the
    percentage overlap area with ROI after sorting

    Parameters
    ----------
    products: geopandas.geodataframe.GeoDataFrame
        returned from Sentinelsat api
    ROI: shapely.geometry.multipolygon.MultiPolygon
        Region of interest to check for common geometry
    sorting_params : list of str
        Which parameters to sort by
    sorting_params_ascending : list of bool
        Whether the `sorting_params` should be ascending

    Returns
    -------
    panda dataframe
    """
    column_label = ["Product_id", "overlap_area"]  # overlap_area is the absolute area
    table = []

    for i in range(0, products.shape[0]):
        s_geometry = products["geometry"][i]
        if s_geometry.intersects(ROI):
            common_area = (s_geometry.intersection(ROI)).area
        else:
            common_area = 0

        data = [products.index[i], common_area]
        table.append(data)

    dataframe = pd.DataFrame(table, columns=column_label)
    sorted_dataframe = dataframe.sort_values(by=["overlap_area"], ascending=False)

    # Rearranging products as per the "Percent_overlap"
    products_sorted = products.reindex(sorted_dataframe["Product_id"])
    products_sorted["overlap_area"] = list(sorted_dataframe["overlap_area"])
    products_sorted = products_sorted.sort_values(
        by=sorting_params, ascending=sorting_params_ascending
    )
    products_sorted = products_sorted[products_sorted["overlap_area"] > 0.0]
    return products_sorted


def sort_S1(s_products_df, ROI, **kwargs):
    """
    Sort Sentinel tiles based on the common ROI geometry
    Returns panda dataframe with first column as product id and second column as the
    percentage overlap area with ROI after sorting

    Parameters
    ----------
    s_product_df: geopandas.geodataframe.GeoDataFrame returned from Sentinelsat api
    ROI: shapely.geometry.multipolygon.MultiPolygon

    Keyword Arguments
    -----------------
    sorting_params : list of str
        Which parameters to sort by (Default ["overlap_area","beginposition"])
    sorting_params_ascending : list of bool
        Whether the `sorting_params` should be ascending (Default [False, True])

    Returns
    -------
        panda dataframe
    """
    return sort_sentinel_products(
        s_products_df,
        ROI,
        sorting_params=kwargs.get("sorting_params", ["overlap_area", "beginposition"]),
        sorting_params_ascending=kwargs.get("sorting_params_ascending", [False, True]),
    )


def sort_S2(s_products_df, ROI, **kwargs):
    """
    Sort Sentinel2 tiles based on the common ROI geometry
    Returns panda dataframe with first column as product id and second column as the
    percentage overlap area with ROI after sorting

    Parameters
    ----------
    s_product_df: geopandas.geodataframe.GeoDataFrame
        returned from Sentinelsat api
    ROI: shapely.geometry.multipolygon.MultiPolygon
        Region of interest to check for common geometry

    Keyword Arguments
    -----------------
    sorting_params : list of str
        Which parameters to sort by
        Default:  ["overlap_area","cloudcoverpercentage","beginposition"]
    sorting_params_ascending : list of bool
        Whether the `sorting_params` should be ascending
        Default: [False, True, True]

    Returns
    -------
    panda dataframe
    """
    sorting_params = kwargs.get(
        "sorting_params", ["overlap_area", "cloudcoverpercentage", "beginposition"]
    )
    sorting_params_ascending = kwargs.get(
        "sorting_params_ascending", [False, True, True]
    )
    return sort_sentinel_products(
        s_products_df, ROI, sorting_params, sorting_params_ascending
    )


def select_sentinel_products(
    products, ROI, sorting_params, sorting_params_ascending, **kwargs
):
    """
    Function to select Sentinel products based on overlap with
    ROI to cover the complete ROI

    Parameters:
    s_products : sorted geopanda dataframe returned from sentinelsat API
                 based on "Percent_overlap" with ROI
    ROI        : Region of Interest
    print_fig  : Boolean value, If passed True, prints all the figures

    Returns list of Sentinel 1 products to download
    """
    if kwargs.get("print_fig", False):
        fig = sen_plot.plot_ROI(ROI, grid=True)
        display.display(fig)
        display.clear_output(wait=True)

    s_products = products
    # s_products = products.query("Percent_overlap > 0.0")

    # "overlap_area" is the actual area selected in the final selection list
    # while the "Percent_area_covered" is the percentage for the covered area

    column_label = ["Product_id", "overlap_area"]
    s_table = []
    Remaining_ROI = ROI
    iteration = 0
    ROI_table = []

    while Remaining_ROI.area >= 1e-10 and s_products.shape[0] > 0:
        iteration = iteration + 1
        s_geometry = s_products.iloc[0]["geometry"]

        overlap = s_geometry.intersection(Remaining_ROI)

        data = [s_products.index[0], s_products.iloc[0]["overlap_area"]]
        s_table.append(data)

        Remaining_ROI = Remaining_ROI.difference(s_geometry)

        ROI_table.append(overlap)

        if kwargs.get("print_fig", False):
            ax.add_patch(PolygonPatch(overlap, fc="red"))
            if iteration == 1:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())
                display.clear_output(wait=True)

        # Now remove this combination
        # first find index of first row
        row1_index = s_products.index[0]
        s_products = s_products.drop(index=row1_index)
        # Resort the the sentinel products

        s_products = sort_S1(
            s_products,
            Remaining_ROI,
            sorting_params=sorting_params,
            sorting_params_ascending=sorting_params_ascending,
        )

    s_final_df = pd.DataFrame(s_table, columns=column_label)
    s_final_products = products.reindex(s_final_df["Product_id"])
    s_final_products["Percent_area_covered"] = list(
        s_final_df["overlap_area"] / ROI.area * 100
    )

    # n_prod = s_final_products.shape[0]
    # print("Selected Sentinel-1 Products : ", n_prod)
    # print(s_final_products[["beginposition", "Percent_area_covered"]])

    return s_final_products, ROI_table


def select_S1(s1_products_df, ROI, **kwargs):
    """Select Sentinel-1 products based on overlap with ROI to cover the complete ROI.

    Parameters:
    s_products : sorted geopanda dataframe returned from sentinelsat API
                 based on "Percent_overlap" with ROI
    ROI        : Region of Interest
    print_fig  : Boolean value, If passed True, prints all the figures

    Keyword Arguments
    -----------------
    sorting_params : list of str
        Which parameters to sort by (Default ["overlap_area","beginposition"])
    sorting_params_ascending : list of bool
        Whether the `sorting_params` should be ascending (Default [False, True])

    Returns
    -------
    list of Sentinel 1 products to download
    """
    sorting_params = kwargs.get("sorting_params", ["overlap_area", "beginposition"])
    sorting_params_ascending = kwargs.get("sorting_params_ascending", [False, True])

    if kwargs.get("print_fig", False):
        fig = sen_plot.plot_ROI(ROI, grid=True)
        display.display(fig)
        display.clear_output(wait=True)

    products, ROI_table = select_sentinel_products(
        s1_products_df, ROI, sorting_params, sorting_params_ascending
    )

    # n_prod = products.shape[0]
    # print("Selected Sentinel-1 Products : ", n_prod)
    # print(products[["beginposition", "Percent_area_covered"]])

    return products, ROI_table


def select_S2(s2_products_df, ROI, **kwargs):
    """Select Sentinel-2 products based on overlap with ROI to cover the complete ROI.

    Parameters
    ----------
    s_products : sorted geopanda dataframe returned from sentinelsat API
                 based on "Percent_overlap" with ROI
    ROI        : Region of Interest
    print_fig  : Boolean value, If passed True, prints all the figures

    Keyword Arguments
    -----------------
    sorting_params : list of str
        Which parameters to sort by (Default ["overlap_area", "cloudcoverpercentage", "beginposition")
    sorting_params_ascending : list of bool
        Whether the `sorting_params` should be ascending (Default [False, True, True])

    Returns
    -------
    list of Sentinel products to download
    """
    sorting_params = kwargs.get(
        "sorting_params", ["overlap_area", "cloudcoverpercentage", "beginposition"]
    )
    sorting_params_ascending = kwargs.get(
        "sorting_params_ascending", [False, True, True]
    )

    if kwargs.get("print_fig", False):
        fig = sen_plot.plot_ROI(ROI, grid=True)
        display.display(fig)
        display.clear_output(wait=True)

    products, ROI_table = select_sentinel_products(
        s2_products_df, ROI, sorting_params, sorting_params_ascending
    )

    # n_prod = products.shape[0]
    # print("Selected Sentinel-2 Products : ", n_prod)
    # print(products[["beginposition", "Percent_area_covered"]])

    return products, ROI_table


def get_s2products_between_dates(start_date, end_date, geojson, cloud_cover, api):
    ROI_footprint = geojson_to_wkt(geojson)
    s2_products_df = gpd.GeoDataFrame()
    start = start_date
    starts = [
        start_date + timedelta(days=week_no * 7)
        for week_no in range(0, math.ceil((end_date - start_date).days / 7))
    ]
    cloud_cover = (0, 20)
    for start in starts:
        end = (start + timedelta(days=6)).strftime("%Y%m%d")
        start = start.strftime("%Y%m%d")
        _products_df = find_S2(ROI_footprint, start, end, api, cloud_cover=cloud_cover)
        s2_products_df = s2_products_df.append(_products_df)
    return sort_S2(s2_products_df, shape(geojson["geometry"]))


def existing_processed_products():
    """Read CSV of products that have already been processed."""
    filename = Path(SENTINEL_ROOT) / "used-products.csv"
    if not filename.exists():
        return None
    dataset = pd.read_csv(filename)
    return dataset


def has_product_been_used(uuid):
    """Check if this product has been used previously."""
    existing = existing_processed_products()
    if not isinstance(existing, pd.DataFrame):
        return False
    has_uuid = not existing.query("uuid == @uuid").empty
    return has_uuid


def mark_product_as_used(*, s1_uuid, s1_date, s2_uuid, s2_date, collocated_folder):
    """Add information about this product to the global 'used product' tracker.

    Arguments MUST be passed as keywords (e.g. uuid=<some_uuid_variable>).
    """
    existing_products = existing_processed_products()
    print("In function mark_product_as_used")
    if not isinstance(existing_products, pd.DataFrame):
        existing_products = pd.DataFrame()
    row = {
        "Processed-date": date.today(),
        "S1-uuid": s1_uuid,
        "S1-date": s1_date,
        "S2-uuid": s2_uuid,
        "S2-date": s2_date,
        "Collocated-folder": collocated_folder,
    }
    existing_products = existing_products.append(row, ignore_index=True)
    filename = Path(SENTINEL_ROOT) / "used-products.csv"
    existing_products.to_csv(filename, index=False)

def download_S2_GCS_py(s2_product, credentials_file):
    """If Sentinel-2 L2A Data has arleady been archived on the sentinel hub, this function
    downloads the data from the Google Cloud Server. SAFE files will be saved in /var/satellite-data/
    Uses the Python API 
    Requires Google Cloud Storage credentials
    export GOOGLE_APPLICATION_CREDENTIALS="credentials_gs.json"


    Argument in:
    s2_products_df from previous sentinel query"""

    
    date = s2_product.beginposition.to_pydatetime()
    year = str(date.year)
    month = str(date.month)
    day = str(date.day)
    productname = s2_product.title
    utm = productname[39:41]
    latb = productname[41:42]
    square = productname[42:44]
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_file)

#    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "demo4_preprocessing/credentials_gs.json"

    client = storage.Client()
    bucket = client.bucket("gcp-public-data-sentinel-2")
    blob_prefix = "L2/tiles/{}/{}/{}/{}.SAFE/".format(utm, latb, square, productname)
    download_from_googlecloud(client, bucket, blob_prefix, productname, SENTINEL_ROOT)


def download_S2_GCS(s2_product):
    """If Sentinel-2 L2A Data has arleady been archived on the sentinel hub, this function
    downloads the data from the Google Cloud Server. SAFE files will be saved in /var/satellite-data/

    Argument in:
    s2_products_df from previous sentinel query"""

    date = s2_product.beginposition.to_pydatetime()
    year = str(date.year)
    month = str(date.month)
    day = str(date.day)
    productname = s2_product.title
    utm = productname[39:41]
    latb = productname[41:42]
    square = productname[42:44]

    # tiles/[UTM code]/latitude band/square/productname.SAFE
    proc_output = subprocess.run(
        [
            "gsutil",
            "-m",
            "cp",
            "-r",
            "gs://gcp-public-data-sentinel-2/L2/tiles/{}/{}/{}/{}.SAFE".format(
                utm, latb, square, productname
            ),
            "{}".format(SENTINEL_ROOT),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        # stderr=subprocess.DEVNULL # hide gpt's info and warning messages
    )


def download_S2_AWS(s2_product):
    """If Sentinel-2 L2A Data has arleady been archived on the sentinel hub, this function
    downloads the data from the AWS. SAFE files will be saved in /var/satellite-data/

    Argument in:
    s2_products_df from previous sentinel query
    """

    date = s2_product.beginposition.to_pydatetime()
    year = str(date.year)
    month = str(date.month)
    day = str(date.day)
    productname = s2_product.title
    utm = productname[39:41]
    latb = productname[41:42]
    square = productname[42:44]

    # tiles/[UTM code]/latitude band/square/[year]/[month]/[day]/[sequence]/DATA

    longstring = (
        "aws s3 cp s3://sentinel-s2-l2a/tiles/"
        + utm
        + "/"
        + latb
        + "/"
        + square
        + "/"
        + year
        + "/"
        + month
        + "/"
        + day
        + "/ "
        + SENTINEL_ROOT
        + productname
        + ".SAFE --request-payer requester --recursive"
    )
    process = subprocess.run(
        longstring,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def download_S2_sentinelhub(s2_product):
    """If Sentinel-2 L2A Data has arleady been archived on the sentinel hub, this function
    downloads the data from the AWS. SAFE files will be saved in /var/satellite-data/

    Argument in:
    s2_products_df from previous sentinel query
    """

    # tiles/[UTM code]/latitude band/square/[year]/[month]/[day]/[sequence]/DATA
    proc_output = subprocess.run(
        [
            "sentinelhub.aws",
            "--product",
            "{}".format(s2_product.title),
            "-f",
            "{}".format(SENTINEL_ROOT),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        # stderr=subprocess.DEVNULL # hide gpt's info and warning messages
    )


def download_S1_AWS(s1_product):
    """If Sentinel-1 GRD Data has arleady been archived on the sentinel hub, this function
    downloads the data from the AWS. SAFE files will be saved in /var/satellite-data/

    Argument in:
    s1_products_df from previous sentinel query
    """

    date = s1_product.beginposition.to_pydatetime()
    year = str(date.year)
    month = str(date.month)
    day = str(date.day)
    productname = s1_product.title

    # [product type]/[year]/[month]/[day]/[mode]/[polarization]/[product identifier]
    longstring = (
        "aws s3 cp s3://sentinel-s1-l1c/GRD/"
        + year
        + "/"
        + month
        + "/"
        + day
        + "/IW/DV/"
        + productname
        + "/ "
        + SENTINEL_ROOT
        + productname
        + ".SAFE --request-payer requester --recursive"
    )
    process = subprocess.run(
        longstring,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def download_S1_NOAA_py(s1_product, auth=None):
    """If Sentinel-1 GRD Data has arleady been archived on the sentinel hub, this function
    downloads the data from ASF NASA API. ZIP files will be saved
	
    Uses urllib instead of wget

    Argument in:
    s1_products_df from previous sentinel query
    """

    if auth:
        username = auth["username"]
        password = auth["password"]
        outputpath = SENTINEL_ROOT
    else:
        username = input("Earthdata Login Username: ")
        password = getpass.getpass(prompt="Password: ")
        outputpath = input(
            "Where would you like to save the downlaoded data? (e.g /media/raid/satellite-data/ ) "
        )

    productname = s1_product.title
    producttype = productname[7:10]
    satellite = productname[2]
 
    
    url = "https://datapool.asf.alaska.edu/{}_HD/S{}/{}.zip".format(producttype, satellite, productname)
    
    with requests.Session() as s:
        s.auth = (username, password)
        r1 = s.request('get', url)
        r = s.get(r1.url, auth=(username, password))
        if r.ok:
            with open(outputpath+'/'+url.split('/')[-1], 'wb') as f:
                f.write(r.content)
            f.close()
        s.close()



def download_S1_NOAA(s1_product, auth=None):
    """If Sentinel-1 GRD Data has arleady been archived on the sentinel hub, this function
    downloads the data from ASF NASA API. ZIP files will be saved

    Argument in:
    s1_products_df from previous sentinel query
    """



    if auth:
        username = auth["username"]
        password = auth["password"]
        outputpath = SENTINEL_ROOT
    else:
        username = input("Earthdata Login Username: ")
        password = getpass.getpass(prompt="Password: ")
        outputpath = input(
            "Where would you like to save the downlaoded data? (e.g /media/raid/satellite-data/ ) "
        )

    productname = s1_product.title
    producttype = productname[7:10]
    satellite = productname[2]
   
    args = [
        "wget",
        "-c",
        f"--http-user={username}",
        f"--http-password='{password}'",
        f"https://datapool.asf.alaska.edu/{producttype}_HD/S{satellite}/{productname}.zip",
        "-P",
        outputpath,
        "--quiet",
    ]
    process = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return process.returncode


class CoordinateOutOfBoundsError(Exception):
    """Exception representing known gpt issue 'coordinate out of bounds'."""

    pass


class SentinelPreprocessor:
    """SentinelPreprocessor wraps the Sentinel API with some utility.

    This mainly provides an ability to list products that have not yet been
    downloaded, along with download and preprocess chosen products.
    """

    def __init__(self, config_filename, credentials=None, **kwargs):
        """Init with SentinelAPI credentials.

        Arguments
        ---------
        credentials : str or os.PathLike
            Path to SentinelAPI credentials [default: 'credentials.json']
        config_filename : str or os.PathLike
            Path to SentinelPreprocessor configuration (see `senprep.create_config`)

        Keyword Arguments
        -----------------
        rebuild : bool
            Force rebuilding of products
        """
        if not credentials and Path("credentials.json").exists():
            self.api = load_api("credentials.json")
        elif credentials:
            self.api = load_api(credentials)
        else:
            raise Exception(
                "Either pass credentials file, or have 'credentials.json' in directory"
            )
        self.start = None
        self.end = None
        config = json.load(open(config_filename))
        self.start = config["dates"][0]
        self.end = config["dates"][1]
        self.size = config["size"]
        self.overlap = config["overlap"]
        self.cloudcover = config["cloudcover"]
        self.roi_name = config["name"]
        self.roi = roiutil.ROI(config["geojson"])
        self.bands_S1 = config["bands_S1"]
        self.bands_S2 = config["bands_S2"]
        self.n_available = None
        self.required_products = dict()
        self.product_map = []
        self.available_s1 = []
        self.available_s2 = []
        self.ran_list = False
        self.rebuild = kwargs.get("rebuild", False)
        self.primary = kwargs.get("primary", None)
        if self.primary == "S2":
            self.secondary = "S1"
        elif self.primary == "S1":
            self.secondary = "S2"
        self.skip_week = kwargs.get("skip_week", False)
        self.skip_secondary = kwargs.get("skip_secondary", False)
        self.full_collocation = kwargs.get("full_collocation", False)
        self.max_S1_products_per_S2 = (
            1  ## Max no of S1 products to be retained per S2 product
        )
        self.S1_delta_time = 3  ## Search S1 products within (S1_delta_time) days of S2
        self.external_bucket = kwargs.get("external_bucket", False)
        self.available_area = kwargs.get("available_area", False)
        self.mode = kwargs.get("mode", "all")

    def __make_roi_footprint(self, geojson):
        """Workaround to account for the fact that geojson_to_wkt was
        working with read_geojson, which requires a file"""
        f = tempfile.NamedTemporaryFile("w")
        f.write(geojson)
        f.seek(0)
        return geojson_to_wkt(read_geojson(f.name))

    def __repr__(self):
        """Show region information."""
        msg = "SentinelSAT Pre-processor"
        date_msg = "NOT SET"
        roi_msg = "NOT SET"
        available_msg = "NOT SEARCHED YET (run `.find_products()`)"
        date_msg = f"{self.start} till {self.end}"
        roi_msg = f"{self.roi_name}"
        if self.n_available:
            available_msg = f"{self.n_available[0]} S2"
            available_msg += f", {self.n_available[1]} S1"
        msg += f"\n> DATES     | {date_msg}"
        msg += f"\n> ROI       | {roi_msg}"
        msg += f"\n> AVAILABLE | {available_msg}"
        return msg

    def find_primary(self, footprint, start_date, end_date):
        if self.primary == "S2":
            primary_products_df = find_S2(
                footprint, start_date, end_date, self.api, cloud_cover=self.cloudcover
            )
        elif self.primary == "S1":
            primary_products_df = find_S1(
                footprint, start_date, end_date, self.api, plot_tiles=False
            )
        return primary_products_df

    def sort_primary(self, primary_products, footprint):
        if self.primary == "S2":
            primary_products_sorted = sort_S2(primary_products, footprint)
        elif self.primary == "S1":
            primary_products_sorted = sort_S1(primary_products, footprint)
        return primary_products_sorted

    def select_primary(self, primary_products_sorted, footprint, print_fig):
        if self.primary == "S2":
            primary_final_df, ROI_table_primary = select_S2(
                primary_products_sorted, footprint
            )
        elif self.primary == "S1":
            primary_final_df, ROI_table_primary = select_S1(
                primary_products_sorted, footprint
            )
        return primary_final_df, ROI_table_primary

    def find_secondary(self, primary_prod, roi_primary, plot_tiles):
        centre_date = primary_prod["beginposition"]
        start_date = centre_date - timedelta(days=self.S1_delta_time)
        end_date = centre_date + timedelta(days=self.S1_delta_time)
        if self.primary == "S2":
            secondary_products_df = find_S1(
                roi_primary, start_date, end_date, self.api, plot_tiles=False
            )
        elif self.primary == "S1":
            secondary_products_df = find_S2(
                roi_primary, start_date, end_date, self.api, cloud_cover=self.cloudcover
            )
        return secondary_products_df

    def sort_secondary(self, secondary_products, footprint):
        if self.primary == "S2":
            secondary_products_sorted = sort_S1(secondary_products, footprint)
        elif self.primary == "S1":
            secondary_products_sorted = sort_S2(secondary_products, footprint)
        return secondary_products_sorted

    def select_secondary(
        self, secondary_products_sorted, primary_product, footprint, print_fig=False
    ):
        ## Find out the time difference in hours between primary and secondary

        secondary_products_sorted["abs_time_delta_from_primary_hrs"] = (
            secondary_products_sorted["beginposition"]
            - primary_product["beginposition"]
        ).abs().dt.total_seconds() / 3600

        secondary_products_sorted = secondary_products_sorted.sort_values(
            by=["overlap_area", "abs_time_delta_from_primary_hrs"],
            ascending=(False, True),
        )

        if self.primary == "S2":
            secondary_final_df, ROI_table_secondary = select_S1(
                secondary_products_sorted, footprint
            )
        elif self.primary == "S1":
            secondary_final_df, ROI_table_secondary = select_S2(
                secondary_products_sorted, footprint
            )

        return secondary_final_df, ROI_table_secondary

    def find_products(self):
        """
        Query SentinelAPI for matching products.

        """
        start_date = nearest_previous_monday(yyyymmdd_to_date(self.start))
        end_date = nearest_next_sunday(yyyymmdd_to_date(self.end))

        print("Initial date, Ending date:", start_date, end_date)

        start = start_date
        starts = [
            start_date + timedelta(days=week_no * 7)
            for week_no in range(0, math.ceil((end_date - start_date).days / 7))
        ]

        for start in starts:
            end = (start + timedelta(days=6)).strftime("%Y%m%d")
            start = start.strftime("%Y%m%d")

            print(
                "\n \n-----------------------------------------------------------------------"
            )
            print("week-start, week-end: ", start, "-", end)

            week_product_map = []
            primary_products_df = self.find_primary(self.roi.footprint, start, end)

            if (
                primary_products_df.empty
                and self.skip_week == False
                and self.available_area == False
            ):
                # If we don't pass the skip argument, ask users what they want to do
                # when we don't have any matching products
                _msg = (
                    "No matching {self.master} product found for week {start}..{end}. Choose:\n"
                    + "  y     : skip this week\n"
                    + "  n     : abort the processing\n"
                    + "  y_all : skip all weeks not matching required specification"
                    + "CHOICE> "
                )
                user_input = input(_msg)
                if user_input == "y":
                    continue
                elif user_input == "n":
                    raise Exception("Processing aborted")
                elif user_input == "y_all":
                    self.skip_week = True
                    continue
                else:
                    raise Exception("Invalid input, Processing aborted")

            primary_products_sorted = self.sort_primary(
                primary_products_df, self.roi.shape
            )
            primary_final_df, ROI_table_primary = self.select_primary(
                primary_products_sorted, self.roi.shape, print_fig=False
            )

            Area_covered = primary_final_df["Percent_area_covered"].sum()
            if (
                Area_covered < 99
                and self.skip_week == False
                and self.available_area == False
            ):
                print(
                    f"Complete ROI is not covered by the primary {self.primary} product"
                )
                print(f"Area covered by {self.primary} products is {Area_covered}%")
                user_input = input(
                    f"Press 'y' to skip this week, 'y_all' to skip all weeks not matching required specifications, 'a' to process the available part of ROI, 'n' to abort the processing: "
                )
                if user_input == "y":
                    continue
                elif user_input == "n":
                    logging.debug(
                        f"Complete ROI is not covered by the primary {self.primary} product, Area covered by {self.primary} products is {Area_covered}% Processing aborted"
                    )
                    raise Exception("Processing aborted")
                elif user_input == "y_all":
                    self.skip_week = True
                    continue
                elif user_input == "a":
                    self.available_area = True
                else:
                    raise Exception("Invalid input, Processing aborted")

            primary_fig_title = (
                self.primary + " tiles and ROI from " + start + " to " + end
            )
            #             sen_plot.plot_Stiles_plus_ROI(self.roi.shape, primary_final_df , s_tiles_color="green", grid=False, title =s2_fig_title)

            for i in range(0, primary_final_df.shape[0]):
                primary_prod = primary_final_df.iloc[i]

                if not self.skip_secondary:
                    try:
                        secondary_products_df = self.find_secondary(
                            primary_prod, ROI_table_primary[i], plot_tiles=False
                        )
                        secondary_products_sorted = self.sort_secondary(
                            secondary_products_df, ROI_table_primary[i]
                        )
                        secondary_final_df, ROI_table_secondary = self.select_secondary(
                            secondary_products_sorted,
                            primary_prod,
                            ROI_table_primary[i],
                            print_fig=False,
                        )
                    except sentinel.SentinelAPIError:
                        #                         print('sentinelsat.sentinel.SentinelAPIError')
                        #                         raise Exception('sentinelsat.sentinel.SentinelAPIError JSON')
                        continue
                    #                 secondary_fig_title = self.secondary+'tiles and ROI from '+start+' to '+end
                    #                 sen_plot.plot_S1S2tiles_plus_ROI( ROI_table_primary[i], secondary_final_df ,pd.DataFrame([primary_prod]), grid=False, title=s1_fig_title)
                    if secondary_final_df.empty and not self.skip:
                        _msg = (
                            "No matching {self.secondary} for corresponding {self.primary} found for the week {start} - {end}. Choose:\n"
                            + "  y     : skip this week\n"
                            + "  n     : abort the processing\n"
                            + "  y_all : skip all weeks not matching required specification"
                            + "CHOICE> "
                        )
                        logging.info(_msg)
                        user_input = input(_msg)
                        if user_input == "y":
                            continue
                        elif user_input == "n":
                            raise Exception("Processing aborted")
                        elif user_input == "y_all":
                            self.skip = True
                            continue
                        else:
                            raise Exception("Invalid input, Processing aborted")
                    else:
                        row_no = 0
                        for _, secondary_row in secondary_final_df.iterrows():
                            if self.primary == "S2":
                                s2_prod = primary_prod
                                s1_prod = secondary_row
                            else:
                                s1_prod = primary_prod
                                s2_prod = secondary_row
                            week_product_map.append(
                                (
                                    start,
                                    s1_prod,
                                    s2_prod,
                                    ROI_table_secondary[row_no],
                                    ROI_table_secondary[row_no].area
                                    / self.roi.shape.area
                                    * 100,
                                )
                            )
                            row_no = row_no + 1

                        week_product_map_df = pd.DataFrame(
                            week_product_map,
                            columns=["week_start", "S1", "S2", "ROI", "ROI_area"],
                        )
                else:
                    if self.primary == "S2":
                        s2_prod = primary_prod
                        s1_prod = None
                    else:
                        s1_prod = primary_prod
                        s2_prod = None
                    week_product_map.append(
                        (
                            start,
                            s1_prod,
                            s2_prod,
                            ROI_table_primary[i],
                            ROI_table_primary[i].area / self.roi.shape.area * 100,
                        )
                    )
                    week_product_map_df = pd.DataFrame(
                        week_product_map,
                        columns=["week_start", "S1", "S2", "ROI", "ROI_area"],
                    )
            week_product_map_df = week_product_map_df.sort_values(
                by=["ROI_area"], ascending=(False)
            )
            week_product_map_df["ROI_no"] = [
                n for n in range(1, len(week_product_map_df) + 1)
            ]

            week_product_map_list = week_product_map_df.values.tolist()
            self.product_map.extend(week_product_map_list)
        self.ran_list = True

    def display_available(self):
        """Display available products."""
        if not self.product_map:
            self.find_products()
        print("-----------------------------------------------------------------------")
        print("Summary")
        print("-----------------------------------------------------------------------")
        print(f"Primary product: {self.primary}")

        for (week, s1_df, s2_df, _, roi_area, roi_no) in self.product_map:
            print(
                f"Week start: {week} | ROI no {roi_no} | Area coverage: {roi_area:.2f}%"
            )
            if s1_df is not None:
                print(
                    f"S1 - {s1_df.uuid} {s1_df.beginposition} {s1_df.Percent_area_covered:.2f}"
                )
            if s2_df is not None:
                print(
                    f"S2 - {s2_df.uuid} {s2_df.beginposition} {s2_df.Percent_area_covered:.2f}"
                )
            print()

    def collocate(self, dir_out_for_roi, ROI_subset, s1, s2):
        """Collocate Sen1 and Sen2 products."""
        s1_title = s1["title"]
        s1_id = s1["uuid"]
        s1_date = s1.beginposition.strftime("%Y%m%d")
        s1_zip = str(Path(SENTINEL_ROOT) / f"{s1_title}.zip")
        if Path(s1_zip).exists():
            print("S1 Zip file exists")
        else:
            s1_zip = str(Path(SENTINEL_ROOT) / f"{s1_title}.SAFE/")
            if Path(s1_zip).exists():
                print("S1 Safe file exists")
            else:
                print("S1 File does not exist")

        s2_title = s2["title"]
        s2_id = s2["uuid"]
        s2_date = s2.beginposition.strftime("%Y%m%d")
        s2_zip = str(Path(SENTINEL_ROOT) / f"{s2_title}.zip")
        if Path(s2_zip).exists():
            print("S2 Zip file exists")
        else:
            s2_zip = str(Path(SENTINEL_ROOT) / f"{s2_title}.SAFE")
            if Path(s2_zip).exists():
                print("S2 Safe file exists")
            else:
                print("S2 File does not exist")

        imagename = f"S1_{s1_id}_S2_{s2_id}.tif"
        filename_s1_collocated = dir_out_for_roi / "S1" / "Collocated" / imagename
        filename_s2_collocated = dir_out_for_roi / "S2" / "Collocated" / imagename

        filename_s1_collocated.parent.mkdir(exist_ok=True, parents=True)
        filename_s2_collocated.parent.mkdir(exist_ok=True, parents=True)

        # Combining the bands_S1 and bands_S2 in a string to pass it to the gpt file
        separator = ","
        bands_S1_string = separator.join(self.bands_S1)
        bands_S2_string = separator.join(self.bands_S2)

        has_files = filename_s1_collocated.exists() and filename_s2_collocated.exists()
        if has_files and not self.rebuild:
            print(f"Collocation already done for {s1_id} and {s2_id}")
            return filename_s1_collocated, filename_s2_collocated
        print(
            "Collocating might take a long time (...hours) depending upon the size of the area collocated"
        )
        # gpt complains if LD_LIBRARY_PATH is not set
        # for some reason, this works on jupyter, but not from terminal

        if "LD_LIBRARY_PATH" not in os.environ:
            os.environ["LD_LIBRARY_PATH"] = "."

        if not self.full_collocation:
            ROI_subset_string = str(ROI_subset).replace("POLYGON ", "POLYGON")
            command_arguments = [
                "gpt",
                "gpt_files/gpt_cloud_masks_bands_specified_subset_without_reprojection.xml",
                "-PS1={}".format(s1_zip),
                "-PS2={}".format(s2_zip),
                "-PCollocate_master={}".format(s2_title),
                "-PS1_write_path={}".format(filename_s1_collocated),
                "-PS2_write_path={}".format(filename_s2_collocated),
                "-Pbands_S1={}".format(bands_S1_string),
                "-Pbands_S2={}".format(bands_S2_string),
                "-PROI={}".format(ROI_subset_string),
            ]
        else:
            command_arguments = [
                "gpt",
                "gpt_files/gpt_cloud_masks_bands_specified.xml",
                "-PS1={}".format(s1_zip),
                "-PS2={}".format(s2_zip),
                "-PCollocate_master={}".format(s2_title),
                "-PS1_write_path={}".format(filename_s1_collocated),
                "-PS2_write_path={}".format(filename_s2_collocated),
                "-Pbands_S1={}".format(bands_S1_string),
                "-Pbands_S2={}".format(bands_S2_string),
            ]

        proc_output = subprocess.run(
            command_arguments,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            # stderr=subprocess.DEVNULL # hide gpt's info and warning messages
        )
        err = proc_output.returncode

        if err:
            # print(proc_output.stdout.decode())
            print(proc_output)
            if "out of bounds" in proc_output.stdout.decode():
                err_msg = f"gpt out of bounds error: {s1_id} and {s2_id}: {err}"
                raise CoordinateOutOfBoundsError(err_msg)
            raise Exception("Collocating: gpt return code %s " % (err))

            # Add this product to the global list of used-products
        if self.full_collocation:
            mark_product_as_used(
                s1_uuid=s1_id,
                s1_date=s1_date,
                s2_uuid=s2_id,
                s2_date=s2_date,
                collocated_folder=dir_out_for_roi,
            )
        return filename_s1_collocated, filename_s2_collocated

    def snap_s1(self, dir_out_for_roi, ROI_subset, s1):
        """Collocate Sen1 and Sen2 products."""
        s1_title = s1["title"]
        s1_id = s1["uuid"]
        s1_date = s1.beginposition.strftime("%Y%m%d")
        s1_zip = str(Path(SENTINEL_ROOT) / f"{s1_title}.zip")
        if Path(s1_zip).exists():
            print("S1 Zip file exists")
        else:
            s1_zip = str(Path(SENTINEL_ROOT) / f"{s1_title}.SAFE/")
            if Path(s1_zip).exists():
                print("S1 Safe file exists")
            else:
                print("S1 File does not exist")

        imagename = f"S1_{s1_id}.tif"
        filename_s1_collocated = dir_out_for_roi / "S1" / "Collocated" / imagename
        filename_s1_collocated.parent.mkdir(exist_ok=True, parents=True)

        # Combining the bands_S1 and bands_S2 in a string to pass it to the gpt file
        if "collocationFlags" in self.bands_S1:
            self.bands_S1.remove("collocationFlags")
        separator = ","
        bands_S1_string = separator.join(self.bands_S1).replace("_S", "")

        has_files = filename_s1_collocated.exists()
        if has_files and not self.rebuild:
            print(f"Collocation already done for {s1_id}")
            return filename_s1_collocated
        logging.info("Collocating CAN take hours if the area is large")
        print("Collocating CAN take hours if the area is large")
        # gpt complains if LD_LIBRARY_PATH is not set
        # for some reason, this works on jupyter, but not from terminal

        if "LD_LIBRARY_PATH" not in os.environ:
            os.environ["LD_LIBRARY_PATH"] = "."

        if not self.full_collocation:

            ROI_subset_string = str(ROI_subset).replace("POLYGON ", "POLYGON")

            proc_output = subprocess.run(
                [
                    "gpt",
                    "gpt_files/gpt_cloud_masks_bands_specified_subset_without_reprojection_S1.xml",
                    "-PS1={}".format(s1_zip),
                    "-PS1_write_path={}".format(filename_s1_collocated),
                    "-Pbands_S1={}".format(bands_S1_string),
                    "-PROI={}".format(ROI_subset_string),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                # stderr=subprocess.DEVNULL # hide gpt's info and warning messages
            )

        else:
            proc_output = subprocess.run(
                [
                    "gpt",
                    "gpt_files/gpt_cloud_masks_bands_specified_S1.xml",
                    "-PS1={}".format(s1_zip),
                    "-PS1_write_path={}".format(filename_s1_collocated),
                    "-Pbands_S1={}".format(bands_S1_string),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                # stderr=subprocess.DEVNULL # hide gpt's info and warning messages
            )

        err = proc_output.returncode

        if err:

            # print(proc_output.stdout.decode())
            print(proc_output)
            if "out of bounds" in proc_output.stdout.decode():
                err_msg = f"gpt out of bounds error: {s1_id}: {err}"
                raise CoordinateOutOfBoundsError(err_msg)
            raise Exception("Collocating: gpt return code %s " % (err))
        if self.full_collocation:
            mark_product_as_used(
                s1_uuid=s1_id,
                s1_date=s1_date,
                s2_uuid="None",
                s2_date="None",
                collocated_folder=dir_out_for_roi,
            )

        return filename_s1_collocated

    def snap_s2(self, dir_out_for_roi, ROI_subset, s2):
        """Collocate Sen1 and Sen2 products."""
        s2_title = s2["title"]
        s2_id = s2["uuid"]
        s2_date = s2.beginposition.strftime("%Y%m%d")
        s2_zip = str(Path(SENTINEL_ROOT) / f"{s2_title}.zip")

        if Path(s2_zip).exists():
            print("S2 Zip file exists")
        else:
            s2_zip = str(Path(SENTINEL_ROOT) / f"{s2_title}.SAFE")
            if Path(s2_zip).exists():
                print("S2 Safe file exists")
            else:
                print("S2 File does not exist")
        imagename = f"S2_{s2_id}.tif"
        filename_s2_collocated = dir_out_for_roi / "S2" / "Collocated" / imagename
        filename_s2_collocated.parent.mkdir(exist_ok=True, parents=True)

        # Combining the bands_S2 in a string to pass it to the gpt file
        separator = ","
        bands_S1_string = separator.join(self.bands_S1)
        bands_S2_string = separator.join(self.bands_S2).replace("_M", "")
        logging.debug(
            f"Colloc fn s2 {filename_s2_collocated} exists? {filename_s2_collocated.exists()}"
        )

        has_files = filename_s2_collocated.exists()
        if has_files and not self.rebuild:
            print(f"Collocation already done for {s2_id}")
            return filename_s2_collocated
        print(
            "Collocating might take a long time (...hours) depending upon the size of the area collocated"
        )
        # gpt complains if LD_LIBRARY_PATH is not set
        # for some reason, this works on jupyter, but not from terminal

        if "LD_LIBRARY_PATH" not in os.environ:
            os.environ["LD_LIBRARY_PATH"] = "."

        if not self.full_collocation:
            ROI_subset_string = str(ROI_subset).replace("POLYGON ", "POLYGON")
            command_arguments = [
                "gpt",
                "gpt_files/gpt_cloud_masks_bands_specified_subset_without_reprojection_S2.xml",
                "-PS2={}".format(s2_zip),
                "-PS2_write_path={}".format(filename_s2_collocated),
                "-Pbands_S2={}".format(bands_S2_string),
                "-PROI={}".format(ROI_subset_string),
            ]
        else:
            command_arguments = [
                "gpt",
                "gpt_files/gpt_cloud_masks_bands_specified_S2.xml",
                "-PS2={}".format(s2_zip),
                "-PS2_write_path={}".format(filename_s2_collocated),
                "-Pbands_S2={}".format(bands_S2_string),
            ]
        proc_output = subprocess.run(
            command_arguments,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            # stderr=subprocess.DEVNULL # hide gpt's info and warning messages
        )

        err = proc_output.returncode

        if err:
            # print(proc_output.stdout.decode())
            print(proc_output)
            if "out of bounds" in proc_output.stdout.decode():
                err_msg = f"gpt out of bounds error: {s2_id}: {err}"
                raise CoordinateOutOfBoundsError(err_msg)
            raise Exception("Collocating: gpt return code %s " % (err))
        # Add this product to the global list of used-products
        if self.full_collocation:
            mark_product_as_used(
                s1_uuid="None",
                s1_date="None",
                s2_uuid=s2_id,
                s2_date=s2_date,
                collocated_folder=dir_out_for_roi,
            )
        return filename_s2_collocated

    def crop(
        self, dir_out_for_roi, s1_or_s2, product_id, path_collocated, ROI_subset, roi_no
    ):
        s1_or_s2 = s1_or_s2.upper()
        assert s1_or_s2 in ["S1", "S2"], "s1_or_s2 must be 'S1' or 'S2'"

        if not path_collocated:
            err_msg = (
                f"No {s1_or_s2} collocation file for {product_id}."
                + f"No products OR issue with {s1_or_s2} products"
            )
            print(err_msg)
            return None

        roi_path = str(dir_out_for_roi / f"ROI{roi_no}.geojson")
        raster = rio.open(path_collocated)

        if not path_collocated:
            err_msg = (
                f"No {s1_or_s2} collocation file for {product_id}."
                + f"No products OR issue with {s1_or_s2} products"
            )
            print(err_msg)
            return None

        #         print('ROI_subset',ROI_subset)
        # Don't use 'init' keyword, as it's deprecated
        wgs84 = pyproj.Proj(init="epsg:4326")
        utm = pyproj.Proj(init=str(raster.crs))
        project = partial(pyproj.transform, wgs84, utm)
        utm_ROI = transform(project, ROI_subset)
        #         utm_ROI = utm_ROI.intersection(
        #             utm_ROI
        #         )  ##Just a way around make multipolygon to polygon
        if not hasattr(utm_ROI, "exterior"):
            print("utm_ROI doesn't have an 'exterior'")
            print(f"Type of utm_ROI: {str(type(utm_ROI))}")
        try:
            ### For polygons exterior.coords exists
            utm_ROI = Polygon(list((utm_ROI.exterior.coords)))
            utm_ROI_m = MultiPolygon([utm_ROI])
        except Exception as E:
            ### For multi polygons exterior.coords does not exist
            area_list = [x.area for x in utm_ROI]
            area_array = np.array(area_list)
            max_area_polygon_no = np.argmax(area_array)
            utm_ROI = utm_ROI[max_area_polygon_no]
            if utm_ROI.is_valid == False:
                utm_ROI = utm_ROI.buffer(0)
            utm_ROI_m = utm_ROI
        ROI_gpd = gpd.GeoDataFrame(utm_ROI_m, crs=str(raster.crs))
        ROI_gpd = ROI_gpd.rename(columns={0: "geometry"})
        # explicitly set it as geometry for the GeoDataFrame
        ROI_gpd.set_geometry(col="geometry", inplace=True)
        ROI_gpd.to_file(roi_path, driver="GeoJSON")

        dir_out_clipped = dir_out_for_roi / s1_or_s2 / "Clipped"

        # # Make directory for the clipped file,
        # # and don't complain if it already exists
        dir_out_clipped.mkdir(exist_ok=True, parents=True)

        filename = "{}_roi{}_{}.tif".format(s1_or_s2, roi_no, product_id)
        clipped_file_path = dir_out_clipped / filename
        if clipped_file_path.exists():
            clipped_file_path.unlink()  # Delete a clipped file if it exists

        ### S1 collocated has datatype of float32 while S2 collocated has datatype of Uint16, but if we dont pass output type to gdal warp, it saves both S1 and S2 clipped file in float32 and so for S2, clipped file has much larger size than collocated file
        ### To check later
        #         if s1_or_s2 == 'S1':
        #             gdal_result = gdal.Warp(str(clipped_file_path),str(path_collocated),cutlineDSName = str(roi_path), cropToCutline=True, dstNodata=999999999.0)
        #         elif s1_or_s2 == 'S2':
        #             gdal_result = gdal.Warp(str(clipped_file_path),str(path_collocated),cutlineDSName = str(roi_path), cropToCutline=True, dstNodata=999999999.0, outputType=gdal.gdalconst.GDT_UInt16)
        gdal_result = gdal.Warp(
            str(clipped_file_path),
            str(path_collocated),
            cutlineDSName=str(roi_path),
            cropToCutline=True,
            dstNodata=999999999.0,
        )
        # CD: is gdal_result used by anything?
        # It seems we save the value, but we never actually read or print it
        gdal_result = None  ## Important to initial gdal writing operations

        gdal.Warp(
            str(clipped_file_path),
            str(path_collocated),
            cutlineDSName=str(roi_path),
            cropToCutline=True,
            dstNodata=999999999.0,
        )

        raster.close()

        return clipped_file_path

    def make_patches(self, dir_out, clip_path, s1_or_s2, s1_id, s2_id):
        """Make smaller (potentially overlapping) patches from a geotiff.

        Arguments
        ---------
        dir_out : pathlib.Path
            Directory for ROI
        clip_path : pathlib.Path
            Filename of cropped sentinel geotiff image
        s1_or_s2 : str
            Either "S1" or "S2"
        s1_id : str
            UUID of the SEN1 product
        s2_id : str
            UUID of the SEN2 product

        Returns
        -------
        NO RETURN
        """

        print("Making ", s1_or_s2, " patches")
        # Convert from pathlib.Path to str
        s1_or_s2 = s1_or_s2.upper()
        assert s1_or_s2 in ["S1", "S2"], "s1_or_s2 must be 'S1' or 'S2'"

        dir_out_for_patches = dir_out / s1_or_s2 / "Patches"

        clip_path = str(clip_path)
        raster = rio.open(clip_path)
        raster_im = raster.read(masked=False)
        res = int(raster.res[0])  # Assuming the resolution in both direction as equal
        gdal_dataset = gdal.Open(clip_path)

        # Create a directory to store the patches
        # CD - TODO convert this function to only take the basic directory
        # then make 'dir_out_for_patches' based on the information we have in the function
        # arguments. This lets us to less 'calculation' all over the place (i.e. make the
        # 'patches' function figure out the specific directory, not whatever other
        # function USES make_patches)
        dir_for_patches = dir_out / s1_or_s2 / "PATCHES"
        dir_for_patches.mkdir(exist_ok=True, parents=True)
        dir_out.mkdir(exist_ok=True, parents=True)
        step_row, step_col = 1 - self.overlap[0], 1 - self.overlap[1]
        row_stride = int(self.size[0] * step_row)
        col_stride = int(self.size[1] * step_col)

        for row_pixel_start in range(
            0, raster_im.shape[1] - self.size[0] + 1, row_stride
        ):
            for column_pixel_start in range(
                0, raster_im.shape[2] - self.size[1] + 1, col_stride
            ):
                row_pixel_end = row_pixel_start + self.size[0] - 1
                column_pixel_end = column_pixel_start + self.size[1] - 1
                # Size is (height, width), as per Priti's code,
                # so display size[1]_size[0] (`width_height`) in filename

                if self.skip_secondary and self.primary == "S1":
                    patch_filename = (
                        f"S1_{s1_id}"
                        + f"_{row_pixel_start}_{column_pixel_start}"
                        + f"_{self.size[1]}x{self.size[0]}.tif"
                    )

                elif self.skip_secondary and self.primary == "S2":
                    patch_filename = (
                        f"S2_{s2_id}"
                        + f"_{row_pixel_start}_{column_pixel_start}"
                        + f"_{self.size[1]}x{self.size[0]}.tif"
                    )

                else:
                    patch_filename = (
                        f"S1_{s1_id}"
                        + f"_S2_{s2_id}"
                        + f"_{row_pixel_start}_{column_pixel_start}"
                        + f"_{self.size[1]}x{self.size[0]}.tif"
                    )

                output_filename = str(dir_for_patches / patch_filename)

                start_x, start_y = raster.xy(row_pixel_start, column_pixel_start)
                start_x = start_x - res / 2
                start_y = start_y + res / 2

                end_x, end_y = raster.xy(row_pixel_end, column_pixel_end)
                end_x = end_x + res / 2
                end_y = end_y - res / 2

                projwin = [start_x, start_y, end_x, end_y]
                gdal.Translate(
                    output_filename, gdal_dataset, format="GTiff", projWin=projwin
                )
        raster.close()

    def download(self):
        """Download available products."""
        print("Downloading started")

        if not self.ran_list:
            print("Haven't searched products yet. Running `.find_products()`")
            self.find_products()

        # If we aren't loading the geojson from the region's dir, copy it there
        # if not 'SENTINEL_STORAGE_PATH' in os.environ:
        #     msg = "SENTINEL_STORAGE_PATH must be set using environment variable."
        #     msg += "\nCan also set using os.environ['SENTINEL_STORAGE_PATH'] = '...path...'."
        #     raise Exception(msg)
        # SENTINEL_STORAGE_PATH = os.environ['SENTINEL_STORAGE_PATH']
        parent_dir = Path(SENTINEL_STORAGE_PATH) / self.roi_name
        parent_dir.mkdir(exist_ok=True, parents=True)

        # CD TODO - tidy this up to try and remove the need for so much 'if-else' if possible
        for _, s1, s2, _, _, _ in self.product_map:
            if s1 is not None:  # Download Sentinel 1 Product
                if not self.external_bucket:
                    self.api.download(
                        s1["uuid"], directory_path=SENTINEL_ROOT, checksum=True
                    )
                else:
                    #                     if self.api.get_product_odata(s1["uuid"])['Online']==True:
                    #                         print('\nDownloading', s1["uuid"],'from sentinelsat')
                    #                         self.api.download(s1["uuid"], directory_path=SENTINEL_ROOT, checksum=True)
                    #                     else:
                    print("\nDownloading", s1["uuid"], "from AWS")
                    download_S1_AWS(s1)

            if s2 is not None:  # Download Sentinel 2 Product
                if not self.external_bucket:
                    self.api.download(
                        s2["uuid"], directory_path=SENTINEL_ROOT, checksum=True
                    )
                else:
                    #                     if self.api.get_product_odata(s2["uuid"])['Online']== True:
                    #                         print('\nDownloading', s2["uuid"],'from sentinelsat')
                    #                         self.api.download(s2["uuid"], directory_path=SENTINEL_ROOT, checksum=True)
                    #                     else:
                    #                         print('\nDownloading', s2["uuid"],'from sentinelhub')
                    #                         download_S2_sentinelhub(s2)
                    print("\nDownloading", s2["uuid"], "from Google")
                    download_S2_GCS(s2)

    def process_new(self):
        """Download and preprocess available products."""
        logging.info("Preprocessing started")

        Processing_start_time = time.time()
        if not self.ran_list:
            print("Haven't searched products yet. Running `.find_products()`")
            self.find_products()

        # If we aren't loading the geojson from the region's dir, copy it there
        # if not 'SENTINEL_STORAGE_PATH' in os.environ:
        #     msg = "SENTINEL_STORAGE_PATH must be set using environment variable."
        #     msg += "\nCan also set using os.environ['SENTINEL_STORAGE_PATH'] = '...path...'."
        #     raise Exception(msg)
        # SENTINEL_STORAGE_PATH = os.environ['SENTINEL_STORAGE_PATH']
        parent_dir = Path(SENTINEL_STORAGE_PATH) / self.roi_name
        parent_dir.mkdir(exist_ok=True, parents=True)

        for _, s1, s2, _, _, _ in self.product_map:
            if s1 is not None:  # Download Sentinel 1 Product
                if not self.external_bucket:
                    self.api.download(
                        s1["uuid"], directory_path=SENTINEL_ROOT, checksum=True
                    )
                else:

                    if self.api.get_product_odata(s1["uuid"])["Online"] == True:
                        self.api.download(
                            s1["uuid"], directory_path=SENTINEL_ROOT, checksum=True
                        )
                    else:
                        print("\nDownloading", s1["uuid"], "from AWS")
                        download_S1_AWS(s1)

            if s2 is not None:  # Download Sentinel 2 Product
                if not self.external_bucket:
                    self.api.download(
                        s2["uuid"], directory_path=SENTINEL_ROOT, checksum=True
                    )
                else:
                    if self.api.get_product_odata(s2["uuid"])["Online"] == True:
                        self.api.download(
                            s2["uuid"], directory_path=SENTINEL_ROOT, checksum=True
                        )
                    else:
                        print("\nDownloading", s2["uuid"], "from sentinelhub")
                        download_S2_sentinelhub(s2)

    def process(self):
        """Download and preprocess available products."""
        logging.info("Preprocessing started")

        Processing_start_time = time.time()
        if not self.ran_list:
            print("Haven't searched products yet. Running `.find_products()`")
            self.find_products()
        print(
            "\n \n-----------------------------------------------------------------------"
        )
        print("-----------------------------------------------------------------------")
        print("Products processing started")

        # If we aren't loading the geojson from the region's dir, copy it there
        # if not 'SENTINEL_STORAGE_PATH' in os.environ:
        #     msg = "SENTINEL_STORAGE_PATH must be set using environment variable."
        #     msg += "\nCan also set using os.environ['SENTINEL_STORAGE_PATH'] = '...path...'."
        #     raise Exception(msg)
        # SENTINEL_STORAGE_PATH = os.environ['SENTINEL_STORAGE_PATH']
        parent_dir = Path(SENTINEL_STORAGE_PATH) / self.roi_name
        parent_dir.mkdir(exist_ok=True, parents=True)
        # Previously, copied the geojson into the region's output folder
        # but using CT's suggestion, may be possible/wise to remove this code
        # filename_geojson = self.roi["filename"]
        # if not Path(filename_geojson).parent == parent_dir:
        #     shutil.copy(filename_geojson, parent_dir)

        s1_s2_products_existing = existing_processed_products()
        s1_s2_products_existing = s1_s2_products_existing.sort_values(
            by=["Processed-date"], ascending=(False)
        )
        if self.rebuild:
            logging.info("Rebuilding products")
            s1_s2_products_existing = pd.DataFrame()

        for _, s1, s2, ROI_subset, _, roi_no in self.product_map:
            if self.primary == "S2":
                nearest_monday = nearest_previous_monday(s2.beginposition)
            else:
                nearest_monday = nearest_previous_monday(s1.beginposition)
            dir_out_for_roi = (
                Path(SENTINEL_STORAGE_PATH)
                / self.roi_name
                / nearest_monday.strftime("%Y%m%d")
                / f"ROI{roi_no}"
            )
            print("ROI Dir", dir_out_for_roi)
            processing_one_roi = time.time()

            if s1 is not None:
                s1_id = s1["uuid"]
                logging.info(f"- S1 {s1_id}")
                s1_date = s1.beginposition.strftime("%Y%m%d")
                s1_title = s1["title"]

            if s2 is not None:
                s2_id = s2.uuid
                logging.info(f"Processing ROI subset {ROI_subset}, S2 {s2_id}")
                s2_date = s2.beginposition.strftime("%Y%m%d")
                # if has_product_been_used(s2_id):
                #     print(f"Skipping used S2 product {s2_id}")
                #     continue
                s2_title = s2.title

            products_exist = False
            if not self.rebuild:
                path_s1_collocated, path_s2_collocated = None, None

                if not (s1 is None or s2 is None):
                    ## Check whether the collocated products exist
                    if isinstance(s1_s2_products_existing, pd.DataFrame):
                        existing_products = s1_s2_products_existing[
                            (s1_s2_products_existing["S1-uuid"] == s1_id)
                            & (s1_s2_products_existing["S2-uuid"] == s2_id)
                        ]
                        if len(existing_products) > 0:
                            ## sort the products to use the latest collocated product
                            existing_products = existing_products.sort_values(
                                by=["Processed-date"], ascending=(False)
                            )
                            selected_used_product = existing_products.iloc[0]
                            existing_collocated_path = selected_used_product[
                                "Collocated-folder"
                            ]

                            imagename = f"S1_{s1_id}_S2_{s2_id}.tif"
                            path_s1_collocated = (
                                Path(existing_collocated_path)
                                / "S1"
                                / "Collocated"
                                / imagename
                            )
                            path_s2_collocated = (
                                Path(existing_collocated_path)
                                / "S2"
                                / "Collocated"
                                / imagename
                            )

                            if (
                                path_s1_collocated.exists()
                                and path_s2_collocated.exists()
                            ):
                                products_exist = True
                                print(
                                    "S1 and S2 products exist in earlier used-products, So collocation is skipped"
                                )
                                filename_s1_collocated = (
                                    dir_out_for_roi / "S1" / "Collocated" / imagename
                                )
                                filename_s2_collocated = (
                                    dir_out_for_roi / "S2" / "Collocated" / imagename
                                )

                                filename_s1_collocated.parent.mkdir(
                                    exist_ok=True, parents=True
                                )
                                filename_s2_collocated.parent.mkdir(
                                    exist_ok=True, parents=True
                                )
                elif s1 is not None:
                    ## Check whether the processed S1 products exist
                    if isinstance(s1_s2_products_existing, pd.DataFrame):
                        existing_products = s1_s2_products_existing[
                            (s1_s2_products_existing["S1-uuid"] == s1_id)
                        ]
                        if len(existing_products) > 0:
                            ## sort the products to use the latest collocated product
                            existing_products = existing_products.sort_values(
                                by=["Processed-date"], ascending=(False)
                            )
                            selected_used_product = existing_products.iloc[0]
                            existing_collocated_path = selected_used_product[
                                "Collocated-folder"
                            ]
                            if selected_used_product["S2-date"] == "None":
                                imagename = f"S1_{s1_id}.tif"
                            else:
                                s2_id_prev = selected_used_product["S2-uuid"]
                                imagename = f"S1_{s1_id}_S2_{s2_id_prev}.tif"
                            path_s1_collocated = (
                                Path(existing_collocated_path)
                                / "S1"
                                / "Collocated"
                                / imagename
                            )
                            if path_s1_collocated.exists():
                                products_exist = True
                                print(
                                    "S1 product exists in earlier used-products, So snap-processing is skipped"
                                )
                                filename_s1_collocated = (
                                    dir_out_for_roi / "S1" / "Collocated" / imagename
                                )
                                filename_s1_collocated.parent.mkdir(
                                    exist_ok=True, parents=True
                                )

                elif s2 is not None:
                    if isinstance(s1_s2_products_existing, pd.DataFrame):
                        existing_products = s1_s2_products_existing[
                            (s1_s2_products_existing["S2-uuid"] == s2_id)
                        ]
                        if len(existing_products) > 0:
                            ## sort the products to use the latest collocated product
                            existing_products = existing_products.sort_values(
                                by=["Processed-date"], ascending=(False)
                            )
                            selected_used_product = existing_products.iloc[0]
                            existing_collocated_path = selected_used_product[
                                "Collocated-folder"
                            ]
                            if selected_used_product["S1-date"] == "None":
                                imagename = f"S2_{s2_id}.tif"
                            else:
                                s1_id_prev = selected_used_product["S1-uuid"]
                                imagename = f"S1_{s1_id_prev}_S2_{s2_id}.tif"
                            path_s2_collocated = (
                                Path(existing_collocated_path)
                                / "S2"
                                / "Collocated"
                                / imagename
                            )
                            if path_s2_collocated.exists():
                                products_exist = True
                                print(
                                    "S2 product exists in earlier used-products, So snap-processing is skipped"
                                )
                                filename_s2_collocated = (
                                    dir_out_for_roi / "S2" / "Collocated" / imagename
                                )
                                filename_s2_collocated.parent.mkdir(
                                    exist_ok=True, parents=True
                                )

            if (self.rebuild) or ((not self.rebuild) and (not products_exist)):
                try:
                    collocation_start_time = time.time()
                    if (s1 is not None) and (s2 is not None):
                        path_s1_collocated, path_s2_collocated = self.collocate(
                            dir_out_for_roi, ROI_subset, s1, s2
                        )
                    elif s1 is not None:
                        path_s1_collocated = self.snap_s1(
                            dir_out_for_roi, ROI_subset, s1
                        )
                    elif s2 is not None:
                        path_s2_collocated = self.snap_s2(
                            dir_out_for_roi, ROI_subset, s2
                        )

                    print(
                        "Time taken for collocation: ",
                        time.time() - collocation_start_time,
                    )

                except CoordinateOutOfBoundsError as E:
                    # log known bug
                    logging.error(E)
                    continue
                except Exception as E:
                    # log unknown bug
                    logging.error(E)
                    raise E
            # Crop sentinel-1 products
            if s1 is not None:
                if not path_s1_collocated:
                    logging.error(
                        f"No S1 collocation file for {s1_id}, so either no products, or issue with S1 products"
                    )
                    continue

                s1_clip_path = self.crop(
                    dir_out_for_roi,
                    "S1",
                    s1_id,
                    path_s1_collocated,
                    ROI_subset,
                    roi_no,
                )
                if self.skip_secondary:
                    s2_id = None
                try:
                    self.make_patches(dir_out_for_roi, s1_clip_path, "S1", s1_id, s2_id)
                except rio.errors.RasterioIOError:
                    pass

            # Crop sentinel-2 products
            if s2 is not None:
                if not path_s2_collocated:
                    logging.error(
                        f"No S2 collocation file for {s2_id}, so either no products, or issue with S2 products"
                    )
                    continue

                s2_clip_path = self.crop(
                    dir_out_for_roi,
                    "S2",
                    s2_id,
                    path_s2_collocated,
                    ROI_subset,
                    roi_no,
                )
                if self.skip_secondary:
                    s1_id = None
                try:
                    self.make_patches(dir_out_for_roi, s2_clip_path, "S2", s1_id, s2_id)
                except rio.errors.RasterioIOError:
                    pass

            print(
                "Time taken for processing the current ROI:",
                time.time() - processing_one_roi,
            )

        print("Total time taken for processing", time.time() - Processing_start_time)
        logging.info("Preprocessing finished")

    def run(self):
        """Run the pipeline."""
        self.find_products()
        self.display_available()
        if self.mode in ["download", "download_process", "all"]:
            self.download()
        if self.mode in ["process", "download_process", "all"]:
            self.process()


def __main__():
    """Provide a CLI interface for sentinel processing or config creation."""
    from docopt import docopt

    args = docopt(__doc__)

    log_fmt = "%(levelname)s : %(asctime)s : %(message)s"
    log_level = logging.DEBUG
    log_also_to_stderr = False

    config_basename = Path(args["--config"]).name
    # log_filename = f'logs/{config_basename}.log'
    # logging.basicConfig(level=log_level, format=log_fmt, filename=log_filename)
    if log_also_to_stderr:
        logging.getLogger().addHandler(logging.StreamHandler())

    # Use 'mode' and the SentinelProcessor.run() function
    # to prevent duplicate code and make it easier to modify and add more functionality
    # we _always_ do prepper.find_products() and prepper.download()
    # (this happens inside prepper.run())
    if args["create"]:
        # If we want to create a config, just return early
        configutil.create(args["-c"])
        return
    # else...choose what the sentinel preprocessor should do
    elif args["list"]:
        mode = "list"
    elif args["download"]:
        mode = "download"
    elif args["process"]:
        mode = "process"
    elif args["download_process"]:
        mode = "download_process"
    else:
        # Using docopt, this shouldn't be accessible
        # If the appropriate args aren't used, docopt will auto display help
        logging.warning(
            f"Shouldn't be able to reach this branch, due to docopt: args {args}"
        )

    prepper = SentinelPreprocessor(
        config_filename=args["--config"],
        credentials=args["--credentials"],
        rebuild=args["--rebuild"],
        full_collocation=args["--full_collocation"],
        skip_week=args["--skip_week"],
        primary=args["--primary"],
        skip_secondary=args["--skip_secondary"],
        external_bucket=args["--external_bucket"],
        available_area=args["--available_area"],
        mode=mode,
    )
    prepper.run()


if __name__ == "__main__":
    __main__()
