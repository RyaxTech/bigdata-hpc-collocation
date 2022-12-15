#!/usr/bin/env python
"""Prepare sentinel data for a given ROI between dates.

Usage:
    senprep.py create [-c]
    senprep.py list --config CONFIG_FILE --credentials CREDENTIALS_FILE
    senprep.py download --config CONFIG_FILE --credentials CREDENTIALS_FILE [--master MASTER_URL] [--partitions PARTITIONS] [--rebuild]

Options:
    --config CONFIG_FILE             File containing region file to preprocess
    --credentials CREDENTIAL_FILE    File containing sentinel API credentials
    -c                               Let user paste geojson, rather than ask for filename
    --rebuild                        Rebuild products
    --master MASTER_URL	             Spark Master URL
    --partitions PARTITIONS          Number of Spark RDD partitions
"""
import json
import math
import logging
import subprocess
import warnings
import shutil
import os
import tempfile
import pdb
import time

warnings.filterwarnings("ignore", category=FutureWarning)

from datetime import date, timedelta
from functools import partial
from pathlib import Path
# from argparse import ArgumentParser

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import pyproj
import rasterio as rio
import pyperclip
from docopt import docopt
from IPython import display
from descartes.patch import PolygonPatch
from osgeo import gdal
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt, sentinel
from shapely.geometry import Polygon, MultiPolygon, shape
from shapely.ops import transform

# modules local to this src dir
import configutil, roiutil
import pyspark
from pyspark.sql.types import *

SENTINEL_ROOT="/data/app/results/satellite-data/"
SENTINEL_STORAGE_PATH="/data/app/results/satellite-data/Sentinel_Patches/"
#SENTINEL_ROOT = "/mnt/lustre/scratch/nikela/satellite-data/"
#SENTINEL_STORAGE_PATH = "/mnt/lustre/scratch/nikela/satellite-data/Sentinel_Patches/"
GPT_FILES_PATH="/data/app/gpt_files"
DEBUG = False



def nearest_previous_monday(date):
    """Get the Monday before or on a given date.

    Weekday is from 0 (monday) to 6 (sunday)
    so subtract weekday from given date to find the nearest earlier Monday.

    If the passed date IS a Monday, this will be a noop
    i.e. it'll return the same date.
    """
    return date - timedelta(days=date.weekday())


def yyyymmdd_to_date(d):
    yyyy = int(d[:4])
    mm = int(d[4:6])
    dd = int(d[6:8])
    return date(yyyy, mm, dd)


def load_api(credentials_json_file_path):
    """
    Loads the api with the credentials.json file for using with the sentinelsat api
    """
    credentials = json.load(open(credentials_json_file_path))

    api = SentinelAPI(
        credentials["username"],
        credentials["password"],
        credentials["sentinel_url_endpoint"],
    )
    return api


def load_ROI(ROI_path):
    """
    Loads ROI in a shapely geometry
    Parameters:
    ROI_path: Path to the geojson file
    Returns shapely geometry
    """
    with open(ROI_path) as f:
        Features = json.load(f)["features"]

    ## IF ROI is a collection of features
    # ROI =(GeometryCollection([shape(feature["geometry"]) for feature in scotland_features]))

    # IF ROI has a single feature
    for feature in Features:
        ROI = shape(feature["geometry"])

    return ROI


def plot_ROI(ROI, grid=False):
    """
    Plots region of Interest
    Parameters:
    s_product_df: geopandas.geodataframe.GeoDataFrame returned from Sentinelsat api
    ROI: shapely.geometry.multipolygon.MultiPolygon
    """
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.set_title("Region of Interest")
    ax.add_patch(PolygonPatch(ROI, fc="yellow"))

    if grid == True:
        ax.grid(True)
    ax.axis("equal")


def plot_Stiles_plus_ROI(ROI, s_products_df, s_tiles_color="green", grid=False):

    """
    Plots Sentinel tiles along with ROI
    Parameters:
    s_product_df: geopandas.geodataframe.GeoDataFrame returned from Sentinelsat api
    ROI: shapely.geometry.multipolygon.MultiPolygon

    """
    # S1 or S2 tiles

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)

    for i in range(0, s_products_df.shape[0]):
        geometry = s_products_df.iloc[i]["geometry"]
        ax.add_patch(PolygonPatch(geometry, fc=s_tiles_color))
        ax.set_title("Sentinel tiles and ROI")

    if grid == True:
        ax.grid(True)

    ax.add_patch(PolygonPatch(ROI, fc="yellow"))
    ax.axis("equal")


def plot_S1S2tiles_plus_ROI(ROI, s1_products_df, s2_products_df, grid=False):
    """
    Plots Sentinel-1 and Sentinel-2 tiles along with ROI
    Parameters:
    s1_product_df: geopandas.geodataframe.GeoDataFrame returned from Sentinelsat api
    s2_product_df: geopandas.geodataframe.GeoDataFrame returned from Sentinelsat api
    ROI: shapely.geometry.multipolygon.MultiPolygon

    """
    # S1 or S2 tiles

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)

    for i in range(0, s1_products_df.shape[0]):
        geometry = s1_products_df.iloc[i]["geometry"]
        ax.add_patch(PolygonPatch(geometry, fc="blue"))

    for j in range(0, s2_products_df.shape[0]):
        geometry = s2_products_df.iloc[j]["geometry"]
        ax.add_patch(PolygonPatch(geometry, fc="green"))
        ax.set_title("Sentinel tiles and ROI")

    if grid == True:
        ax.grid(True)

    ax.add_patch(PolygonPatch(ROI, fc="yellow"))
    ax.axis("equal")


def sort_S(s_products_df, ROI):
    """
    Sort Sentinel tiles based on the common ROI geometry
    Returns panda dataframe with first column as product id and second column as the
    percentage overlap area with ROI after sorting

    Parameters:
    s_product_df: geopandas.geodataframe.GeoDataFrame returned from Sentinelsat api
    ROI: shapely.geometry.multipolygon.MultiPolygon
    Returns panda dataframe
    """
    column_label = ["Product_id", "Percent_overlap"]
    table = []
    #     sorted_s_products_df=s_products_df.copy()

    for i in range(0, s_products_df.shape[0]):

        s_geometry = s_products_df["geometry"][i]
        if s_geometry.intersects(ROI):
            common_area = (s_geometry.intersection(ROI)).area
        else:
            common_area = 0

        common_area_percent = common_area / (ROI.area) * 100

        data = [s_products_df.index[i], common_area_percent]
        table.append(data)

    dataframe = pd.DataFrame(table, columns=column_label)
    sorted_dataframe = dataframe.sort_values(by="Percent_overlap", ascending=False)

    s_products_df_sorted = s_products_df.reindex(sorted_dataframe["Product_id"]) ## Rearranging s_products_df as per the "Percent_overlap"
    s_products_df_sorted["Percent_overlap"] = list(sorted_dataframe["Percent_overlap"])

    return s_products_df_sorted


def select_Sentinel(s_products_df, ROI, print_fig=True):
    """
    Function to select Sentinel products based on overlap with
    ROI to cover the complete ROI

    Parameters:
    s_products : sorted geopanda dataframe returned from sentinelsat API
                 based on "Percent_overlap" with ROI
    ROI: Region of Interest
    Returns list of Sentinel products to download
    """

    if print_fig == True:

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.grid(True)

        ax.add_patch(PolygonPatch(ROI, fc="yellow"))
        ax.axis("equal")
        ax.legend(["ROI"])
        ax.set_ylabel("Latitude (degree)")
        ax.set_xlabel("Longitude (degree)")
        display.display(plt.gcf())
        display.clear_output(wait=True)

    s_products = s_products_df[s_products_df["Percent_overlap"] > 0.0]

    column_label = ["Product_id", "Percent_area_covered"] ## "Percent_area_covered" is the actual area selected in the final selection list
                                                                  ## while the "Percent_overlap" is the intersection of all the sentinel products with the main ROI
    s_table = []

    Remaining_ROI = ROI

    iteration = 0

    ROI_table = []

    while Remaining_ROI.area >= 0 and s_products.shape[0] > 0:
        #         time.sleep(3)
        iteration = iteration + 1
        s_geometry = s_products.iloc[0]["geometry"]

        data = [s_products.index[0], s_products.iloc[0]["Percent_overlap"]]
        s_table.append(data)

        overlap = s_geometry.intersection(Remaining_ROI)
        overlap_area = overlap.area
        Percent_overlap = overlap_area / (ROI.area) * 100
        Remaining_ROI = Remaining_ROI.difference(s_geometry)

        ROI_table.append(overlap)

        if print_fig == True:
            ax.add_patch(PolygonPatch(overlap, fc="red"))
            if iteration == 1:
                ax.legend(("ROI", "S\u2229ROI",))
                display.display(plt.gcf())
                display.clear_output(wait=True)

            else:
                display.display(plt.gcf())
                display.clear_output(wait=True)

        # Now remove this combination
        # first find index of first row
        row1_index = s_products.index[0]
        s_products = s_products.drop(index=row1_index)

        ## Resort the list
        for i in range(0, s_products.shape[0]):

            s_geometry = s_products.iloc[i]["geometry"]

            if s_geometry.intersects(Remaining_ROI):
                overlap_area = (s_geometry.intersection(Remaining_ROI)).area
                Percent_overlap = overlap_area / (ROI.area) * 100

            else:
                Percent_overlap = 0
            s_products.iloc[i, -1] = Percent_overlap
        #             s_products.iloc[i]["Percent_overlap"]=Percent_overlap

        s_products = s_products.sort_values(by="Percent_overlap", ascending=False)
        s_products = s_products[s_products["Percent_overlap"] > 0.0]

    s_final_df = pd.DataFrame(s_table, columns=column_label)

    s_final_products = s_products_df.reindex(s_final_df["Product_id"])
    s_final_products["Percent_area_covered"] = list(s_final_df["Percent_area_covered"])

    return s_final_products, ROI_table


def find_S1(s2_df, ROI_footprint, api, delta_time, **kwargs):
    """
    Finds S1 products given Region of Interest and corresponding Sentinel2 product
    geopanda dataframe
    Parameters:
    s2_df: Takes geopanda dataframe returned from Sentinelsat api or sorted version of it
    ROI: Region of Interest as shapely geometry

    Key-word Arguments
    ------------------
    plot_tiles : bool [default: True]
        Whether to plot S1 product tiles
    verbose : bool [default: False]
        Whether to display information messages

    Returns S1 products as a geopanda dataframe
    """
    centre_date = s2_df["beginposition"]
    start_date = centre_date - timedelta(days=delta_time)
    end_date = centre_date + timedelta(days=delta_time)
    date = (start_date, end_date)

    s1_products = api.query(
        ROI_footprint, date=date, platformname="Sentinel-1", producttype="GRD"
    )

    s1_products_df = api.to_geodataframe(s1_products)

    if kwargs.get("plot_tiles", True):
        plot_Stiles_plus_ROI(ROI_footprint, s1_products_df, "blue", grid=False)

    Percent_S2_overlap_covered = []
    for i in range(0, s1_products_df.shape[0]):
        overlap = (s1_products_df.iloc[i]["geometry"]).intersection(ROI_footprint)
        Percent_overlap = (overlap.area) / ROI_footprint.area * 100
        Percent_S2_overlap_covered.append(Percent_overlap)

    s1_products_df["Percent_S2_overlap_covered"] = Percent_S2_overlap_covered

    if kwargs.get('verbose', False):
        logging.info("Matching Sentinel-1 Products Found: ", s1_products_df.shape[0])

    print("Matching Sentinel-1 Products Found: ", s1_products_df.shape[0])
    return s1_products_df


def find_S2(ROI_footprint, start_date, end_date, cloud_cover=(0, 20), api=None, **kwargs):
    """Finds S2 products given a Region of Interest.

    Arguments
    ---------
    ROI : shapely.shape
        Region of Interest as shapely geometry
    start_date: str
        format yyyymmdd e.g. 20200601 for 1st June 2020
    end_date: str
        format yyyymmdd e.g. 20200601 for 1st June 2020
    cloud_cover : tuple (decimal, decimal)
        indicating start and end for cloudcover percentage

    Keyword-Arguments
    -----------------
    verbose : bool [default: False]
        Whether to print information messages

    Returns
    -------
    s2_products_df : geopandas.DataFrame
    """
    date = (start_date, end_date)


    s2_products = api.query(
        ROI_footprint,
        date=date,
        platformname="Sentinel-2",
        cloudcoverpercentage=cloud_cover,
        producttype="S2MSI2A",
    )

    s2_products_df = api.to_geodataframe(s2_products)
    n_prod = s2_products_df.shape[0]
    if kwargs.get('verbose', False):
        logging.info(f"Matching Sentinel-2 Products Found ({start_date} to {end_date}): {n_prod}")

    print("Matching Sentinel-2 Products Found",start_date, "to",end_date,': ',n_prod)
    return s2_products_df


def get_s2products_between_dates(start_date, end_date, geojson, cloud_cover, api):
    ROI_footprint = geojson_to_wkt(geojson)
    s2_products_df = gpd.GeoDataFrame()
    start = start_date
    starts = [
        start_date + timedelta(days=week_no * 7)
        for week_no in range(0, math.ceil((end_date - start_date).days / 7))
    ]
    for start in starts:
        end = (start + timedelta(days=6)).strftime("%Y%m%d")
        start = start.strftime("%Y%m%d")

        cloudcoverpercentage = (0, 20)
        _products_df = find_S2(ROI_footprint, start, end, cloudcoverpercentage, api)
        s2_products_df = s2_products_df.append(_products_df)
    return sort_S(s2_products_df, shape(geojson['geometry']))


def existing_processed_products():
    filename = Path(SENTINEL_ROOT) / "used-products.csv"
    if not filename.exists():
        return None
    dataset = pd.read_csv(filename)
    return dataset


def has_product_been_used(uuid):
    """Check if this product has been used previously."""
    existing = existing_processed_products()
    if not isinstance(existing, pd.DataFrame):
        logging.info(f"No products tracked yet. CSV of tracked products doesn't exist.")
        return False
    return existing[existing.uuid == uuid].shape[0] > 0


def mark_product_as_used(*, uuid, product_type, date):
    """Add information about this product to the global 'used product' tracker.

    Arguments MUST be passed as keywords (e.g. uuid=<some_uuid_variable>).
    """
    existing_products = existing_processed_products()
    if not isinstance(existing_products, pd.DataFrame):
        existing_products = pd.DataFrame()
    row = {
        'product': product_type,
        'uuid': uuid,
        'date': date
    }
    existing_products = existing_products.append(row, ignore_index=True)
    filename = Path(SENTINEL_ROOT) / "used-products.csv"
    existing_products.to_csv(filename, index=False)

def make_patches_f(dir_out, clip_path, s1_or_s2, s1_id, s2_id, size, overlap):
    """Make smaller (potentially overlapping) patches from a geotiff.

        Arguments
        ---------
        dir_out : pathlib.Path
            Directory to save patches
        clip_path : pathlib.Path
            Filename of cropped sentinel geotiff image
        s1_or_s2 : str
            Either "S1" or "S2"
        s1_id : str
            UUID of the SEN1 product
        s2_id : str
            UUID of the SEN2 product
        size:
        overlap:
        Returns
        -------
        NO RETURN
    """
    # Convert from pathlib.Path to str
    s1_or_s2 = s1_or_s2.upper()
    assert s1_or_s2 in ["S1", "S2"], "s1_or_s2 must be 'S1' or 'S2'"

    clip_path = str(clip_path)
    raster = rio.open(clip_path)
    raster_im = raster.read(masked=False)
    res = int(raster.res[0])  # Assuming the resolution in both direction as equal
    gdal_dataset = gdal.Open(clip_path)

    # Create a directory to store the patches
    dir_out.mkdir(exist_ok=True, parents=True)
    step_row, step_col = 1 - overlap[0], 1 - overlap[1]
    row_stride = int(size[0] / step_row)
    col_stride = int(size[1] / step_col)

    for row_pixel_start in range(0, raster_im.shape[1] - size[0], row_stride):
        for column_pixel_start in range(
                0, raster_im.shape[2] - size[1], col_stride
        ):
            row_pixel_end = row_pixel_start + size[0] - 1
            column_pixel_end = column_pixel_start + size[1] - 1

            # Size is (height, width), as per Priti's code,
            # so display size[1]_size[0] (`width_height`) in filename
            patch_filename = (
                    f"S1_{s1_id}"
                    + f"_S2_{s2_id}"
                    + f"_{row_pixel_start}_{column_pixel_start}"
                    + f"_{size[1]}x{size[0]}.tif"
            )

            output_filename = str(dir_out / patch_filename)

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
    return

def crop(dir_out_for_roi, s1_or_s2, product_id, path_collocated, ROI_subset, roi_no):
    s1_or_s2 = s1_or_s2.upper()
    assert s1_or_s2 in ["S1", "S2"], "s1_or_s2 must be 'S1' or 'S2'"

    roi_path = str(dir_out_for_roi / f"ROI{roi_no}.geojson")

    raster = rio.open(path_collocated)

    # Don't use 'init' keyword, as it's deprecated
    wgs84 = pyproj.Proj(init="epsg:4326")
    utm = pyproj.Proj(init=str(raster.crs))

    project = partial(pyproj.transform, wgs84, utm)
    utm_ROI = transform(project, ROI_subset)

    utm_ROI = utm_ROI.intersection(
        utm_ROI
    )  ##Just a way around make multipolygon to polygon
    if not hasattr(utm_ROI, 'exterior'):
        logging.warning("utm_ROI doesn't have an 'exterior'")
        logging.warning("Type of utm_ROI:", type(utm_ROI))
    try:
        utm_ROI = Polygon(list((utm_ROI.exterior.coords)))
    except Exception as E:
        if DEBUG:
            pdb.set_trace()
        else:
            raise E

    utm_ROI_m = MultiPolygon([utm_ROI])

    ROI_gpd=gpd.GeoDataFrame(utm_ROI_m, crs=str(raster.crs))
    ROI_gpd = ROI_gpd.rename(columns={0: 'geometry'})
    ROI_gpd.set_geometry(col='geometry', inplace=True)
    ROI_gpd.to_file(roi_path, driver='GeoJSON')

    dir_out_clipped = dir_out_for_roi / s1_or_s2 / "Clipped"

    # # Make directory for the clipped file,
    # # and don't complain if it already exists
    dir_out_clipped.mkdir(exist_ok=True, parents=True)

    filename = "{}_roi{}_{}.tif".format(s1_or_s2, roi_no, product_id)
    clipped_file_path = dir_out_clipped / filename
    if clipped_file_path.exists():
        clipped_file_path.unlink()  # Delete a clipped file if it exists

    gdal_result = None
    gdal_result = gdal.Warp(str(clipped_file_path),str(path_collocated),cutlineDSName = str(roi_path), cropToCutline=True, dstNodata=999999999.0)

    gdal_result = None ## Important to initial gdal writing operations

    raster.close()

    return clipped_file_path

def collocate(dir_for_roi, s1_title, s1_id, s2_title, s2_id, rebuild, bands_S1, bands_S2):
    """Collocate Sen1 and Sen2 products."""
    s1_zip = str(Path(SENTINEL_ROOT) / f"{s1_title}.zip")
    s2_zip = str(Path(SENTINEL_ROOT) / f"{s2_title}.zip")

    imagename = f"S1_{s1_id}_S2_{s2_id}.tif"
    filename_s1_collocated = dir_for_roi / "S1" / "Collocated" / imagename
    filename_s2_collocated = dir_for_roi / "S2" / "Collocated" / imagename

    filename_s1_collocated.parent.mkdir(exist_ok=True, parents=True)
    filename_s2_collocated.parent.mkdir(exist_ok=True, parents=True)

    ## Combining the bands_S1 and bands_S2 in a string to pass it to the gpt file
    separator = ','
    bands_S1_string=separator.join(bands_S1)
    bands_S2_string=separator.join(bands_S2)

    logging.debug(f"Colloc fn s1 {filename_s1_collocated} exists? {filename_s1_collocated.exists()}")
    logging.debug(f"Colloc fn s2 {filename_s2_collocated} exists? {filename_s2_collocated.exists()}")

    has_files = filename_s1_collocated.exists() and filename_s2_collocated.exists()
    if has_files and not rebuild:
        logging.info(f"Collocation already done for {s1_id} and {s2_id}")
        print(f"Collocation already done for {s1_id} and {s2_id}")
        return filename_s1_collocated, filename_s2_collocated
    logging.info("Collocating (can take a long time...hours)")
    print("Collocating (can take a long time...hours)")
        # gpt complains if LD_LIBRARY_PATH is not set
        # for some reason, this works on jupyter, but not from terminal
#    if 'LD_LIBRARY_PATH' not in os.environ:
    os.environ['LD_LIBRARY_PATH'] = '.'
    proc_output = subprocess.run(
        [
                "gpt",
                GPT_FILES_PATH+"/gpt_cloud_masks_bands_specified.xml",
                "-PS1={}".format(s1_zip),
                "-PS2={}".format(s2_zip),
                "-PCollocate_master={}".format(s2_title),
                "-PS1_write_path={}".format(filename_s1_collocated),
                "-PS2_write_path={}".format(filename_s2_collocated),
                "-Pbands_S1={}".format(bands_S1_string),
                "-Pbands_S2={}".format(bands_S2_string),
        ],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            # stderr=subprocess.DEVNULL # hide gpt's info and warning messages
    )
    err = proc_output.returncode
    if err:
        print(proc_output.stdout.decode())
        if "out of bounds" in proc_output.stdout.decode():
            logging.debug(f"gpt out of bounds error: {s1_id} and {s2_id}")
            raise CoordinateOutOfBoundsError(err)
        raise Exception("Collocating: gpt return code %s. Logs %s" % (err, proc_output))
    return filename_s1_collocated, filename_s2_collocated


def mapper_f(x, credentials, rebuild, bands_S1, bands_S2, size, overlap):
    api = load_api(credentials)
    s2, s1, ROI_subset, roi_no, roi_name = x
    s2_id = s2[0]
    s2_title = s2[1]
    s2_date = s2[2]
    s2_date_str = s2_date.strftime("%Y%m%d")
    s1_id = s1[0]
    s1_title = s1[1]
    s1_date = s1[2]
    s1_date_str = s1_date.strftime("%Y%m%d")


    dir_out_for_roi = (
            Path(SENTINEL_STORAGE_PATH)
            / roi_name
            / nearest_previous_monday(s2_date).strftime("%Y%m%d")
            / f"ROI{roi_no}"

    )

    dir_out_S1_patches = dir_out_for_roi / "S1" / "Patches"
    dir_out_S2_patches = dir_out_for_roi / "S2" / "Patches"

    #Downloading S2 product
    counter = 0
    while counter < 10:
        try:
            api.download(s2_id, directory_path=SENTINEL_ROOT, checksum=True)
            break
        except(sentinel.SentinelAPIError, FileNotFoundError) as err:
            logging.warning("Fail to access API with error %s", err)
            time.sleep(5 * counter)
            counter +=1
            continue

    logging.info(f"-S1 {s1_id} S2 {s2_id}")

    #Downloading S1
    counter = 0
    while counter < 10:
        try:
            api.download(s1_id, directory_path=SENTINEL_ROOT, checksum=True)
            break
        except(sentinel.SentinelAPIError, FileNotFoundError) as err:
            logging.warning("Fail to access API with error %s", err)
            time.sleep(5 * counter)
            counter +=1
            continue

    #Collocate
    try:
        path_s1_collocated, path_s2_collocated = collocate(
                dir_out_for_roi, s1_title, s1_id, s2_title, s2_id, rebuild, bands_S1, bands_S2
        )
    except CoordinateOutOfBoundsError as E:
        logging.error(E)
    except Exception as E:
        logging.error(E)
        raise E

    s1_clip_path = crop(
        	dir_out_for_roi,
        	"S1",
            s1_id,
            path_s1_collocated,
            ROI_subset,
            roi_no,
	)

    make_patches_f(dir_out_S1_patches, s1_clip_path, "S1", s1_id, s2_id, size, overlap)

    # Add this S1 product to the global list of used-products
    mark_product_as_used(product_type="S1", uuid=s1_id, date=s1_date) ##definite problem with mark_product_as_used, move it out of the download function

    if not path_s2_collocated:
        logging.error(f"No S2 collocation file for {s2_id}, so either no products, or issue with S1 products")
    # Crop sentinel 2 products
    s2_clip_path = crop(
        dir_out_for_roi,
        "S2",
        s2_id,
        path_s2_collocated,
        ROI_subset,
        roi_no,
    )
    make_patches_f(dir_out_S2_patches, s2_clip_path, "S2", s1_id, s2_id, size, overlap)
    # Add this S2 product to the global list of used-products
    mark_product_as_used(product_type="S2", uuid=s2_id, date=s2_date)

    return


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
        if not credentials and Path('credentials.json').exists():
            self.api = load_api('credentials.json')
        elif credentials:
            self.api = load_api(credentials)
        else:
            raise Exception("Either pass credentials file, or have 'credentials.json' in directory")
        self.start = None
        self.end = None
        config = json.load(open(config_filename))
        self.start = config['dates'][0]
        self.end = config['dates'][1]
        self.size = config['size']
        self.overlap = config['overlap']
        self.cloudcover = config['cloudcover']
        self.roi_name = config['name']
        self.roi = roiutil.ROI(config['geojson'])

        if 'bands_S1' not in config.keys() or 'bands_S2' not in config.keys():
            self.bands_S1, self.bands_S2 = configutil.get_default_bands()
        else:
            self.bands_S1=config['bands_S1']
            self.bands_S2=config['bands_S2']
        self.n_available = None
        self.required_products = dict()
        self.product_map = []
        self.full_product_map = []
        self.available_s1 = []
        self.available_s2 = []
        self.ran_list = False
        self.rebuild = kwargs.get('rebuild', False)
        self.max_S1_products_per_S2=2 ## Max no of S1 products to be retained per S2 product
        self.S1_delta_time = 3 ## Search S1 products within (S1_delta_time) days of S2


    def __make_roi_footprint(self, geojson):
        # Workaround to account for the fact that geojson_to_wkt was
        # working with read_geojson, which requires a file
        f = tempfile.NamedTemporaryFile('w')
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


    def find_products(self):
        """
        Query SentinelAPI for matching products.

        """
        week_no = 1

        # ROI_footprint = geojson_to_wkt(self.roi_features)
        # ROI_shape = shape(self.roi_features['geometry'])

        start_date = yyyymmdd_to_date(self.start)
        end_date=yyyymmdd_to_date(self.end)


        print('start_date, end_date:',start_date, end_date)
        s2_products_df = gpd.GeoDataFrame()
        start = start_date
        starts = [
            start_date + timedelta(days=week_no * 7)
            for week_no in range(0, math.ceil((end_date - start_date).days / 7))
        ]


        for start in starts:
            end = (start + timedelta(days=6)).strftime("%Y%m%d")
            start = start.strftime("%Y%m%d")

            print('start, end: ', start, end)

            s2_products_df =find_S2(self.roi.footprint, start, end, self.cloudcover, self.api)
            s2_products_sorted = sort_S(s2_products_df, self.roi.shape)

            if s2_products_sorted.empty:
                logging.info(f"No S2 products with {self.cloudcover} cloud cover")
                return

            existing = existing_processed_products()
            exists = pd.Series([False] * len(s2_products_sorted), index=s2_products_sorted.index)
            if isinstance(existing, pd.DataFrame):
                exists = s2_products_sorted.uuid.isin(existing.uuid)
            s2_products_existing = s2_products_sorted[exists]
            s2_products_nonexisting = s2_products_sorted[~exists]
            if self.rebuild:
                logging.info("Rebuilding products")
                s2_products_existing = pd.DataFrame()
                s2_products_nonexisting = s2_products_sorted

            s2_final_df, ROI_table = select_Sentinel(
            s2_products_nonexisting, self.roi.shape, print_fig=False
            )
            plot_Stiles_plus_ROI(self.roi.shape, s2_final_df , s_tiles_color="green", grid=False)

            print("Selected Sentinel 2 products:",s2_final_df.shape[0] )
            print(s2_final_df[["beginposition","Percent_area_covered"]])
            #print(s2_final_df.keys())##
            self.ROI_table = ROI_table
            total_s2_available = set()
            total_s1_available = set()
            for i in range(0, s2_final_df.shape[0]):
                s2_prod = s2_final_df.iloc[i]
                s1_final_df = find_S1(s2_prod, ROI_table[i], self.api, self.S1_delta_time,plot_tiles=False)

                # Keep S1 which completely overlap with the ROI in S2
                s1_final_df = s1_final_df[s1_final_df["Percent_S2_overlap_covered"] == 100]
                #s1_final_df = s1_final_df[s1_final_df["Percent_S2_overlap_covered"] >= 95

                ## Find out the time difference in hours between s2 and s1
                s1_final_df["abs_time_delta_from_S2"]=  abs((s1_final_df["beginposition"] - s2_prod["beginposition"]).dt.total_seconds()/3600)

                ## First sort the products based on "Percent_S2_overlap_covered" and then by "abs_time_delta_from_S2"
                s1_final_df = s1_final_df.sort_values(by=["Percent_S2_overlap_covered","abs_time_delta_from_S2"], ascending=(False,True))

                ## Limit the maximum no of S1 products
                #s1_final_df = s1_final_df[:self.max_S1_products_per_S2]
                plot_S1S2tiles_plus_ROI( ROI_table[i], s1_final_df, s2_final_df.iloc[[i]], grid=False)
                print(s1_final_df[["Percent_S2_overlap_covered","beginposition","abs_time_delta_from_S2"]])

                if not s1_final_df.empty:
                    total_s2_available.add(s2_prod.uuid)
                    for _, row in s1_final_df.iterrows():
                        # print('\t', row['summary'])
                        total_s1_available.add(row.uuid)
                        self.full_product_map.append((s2_prod, row, ROI_table[i], i+1))
                    self.product_map.append((s2_prod, s1_final_df, ROI_table[i], i + 1))
                else:
                    logging.info(f"S2 product {i} has no matching S1 products")

        self.available_s2 = total_s2_available
        self.available_s1 = total_s1_available
        self.n_available = (len(total_s2_available), len(total_s1_available))
        msg = f"TOTAL: {len(total_s2_available)} S2 and {len(total_s1_available)} S1 unique products available"
        msg += f"\nSkipped {s2_products_existing.shape[0]} already-existing S2 products"
        msg += "\n(and thus their associated s1 products)"
        logging.info(msg)
        self.ran_list = True



    def display_available(self):
        """Display available products."""
        if not self.product_map:
            self.find_products()
        for (s2_df, s1_df, _, _) in self.product_map:
            print("S2 -", s2_df.uuid, 'date',s2_df.beginposition,s2_df.Percent_area_covered, s2_df.title)
            for _, row in s1_df.iterrows():
                print("\tS1 -", row.uuid,row.title,row.beginposition,row.Percent_S2_overlap_covered)

    def create_rdd_list(self):
        """
        Returns a list of all (s1,s2) products to be converted to an RDD
        """
        if not self.ran_list:
            print("Haven't searched products yet. Running `.find_products()`")
            self.find_products()
        toRDD = []
        for e in self.full_product_map:
            ##id, title, date
            s2 = (e[0].uuid, e[0].title, e[0].beginposition)
            ##id, title, date
            s1 = (e[1].uuid, e[1].title, e[1].beginposition)
            ROI_subset = e[2]
            roi_no = e[3]
            roi_name = self.roi_name
            toRDD.append((s2,s1,ROI_subset, roi_no, roi_name))

        return toRDD


def __main__():
    """Provide a CLI interface for sentinel processing or config creation."""
    args = docopt(__doc__)

    log_fmt = '%(levelname)s : %(asctime)s : %(message)s'
    log_level = logging.DEBUG
    log_also_to_stderr = False
    log_dir = os.environ.get("LOG_DIR", "logs")
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    if args['create']:
        logging.basicConfig(level=log_level, format=log_fmt, filename='{log_dir}/config-creation.log')
        if log_also_to_stderr:
            logging.getLogger().addHandler(logging.StreamHandler())
        msg = configutil.create(args['-c'])
        logging.debug(msg)
        return

    config_basename = Path(args['--config']).name
    log_filename = f'{log_dir}/{config_basename}.log'
    logging.basicConfig(level=log_level, format=log_fmt, filename=log_filename)
    if log_also_to_stderr:
        logging.getLogger().addHandler(logging.StreamHandler())
    credentials=args['--credentials']
    prepper = SentinelPreprocessor(
        config_filename=args['--config'],
        credentials=credentials,
        rebuild=args['--rebuild'],
    )
    if args['list']:
        prepper.display_available()
    elif args['download']:
        if args['--master']:
            spark_master = args['--master']
        else:
            spark_master = "local"

        if args['--partitions']: #set number of partitions equal to executors
            partitions = int(args['--partitions'])

        prepper.display_available()
        logging.info("Preprocessing started")

        #Initialize spark session
#	conf = pyspark.SparkConf().setAppName(appName).setMaster(master)
#	spark = pyspark.SparkContext(conf=conf)
        spark = pyspark.sql.SparkSession.builder.appName("SentinelPreprocessing").getOrCreate()
        #Create list to be converted to RDD
        toRDD = prepper.create_rdd_list()
#        partitions = len(toRDD)
        #Broadcast necessary variables
        credentials_bc = spark.sparkContext.broadcast(credentials)
        rebuild_bc = spark.sparkContext.broadcast(prepper.rebuild)
        bands_S1_bc = spark.sparkContext.broadcast(prepper.bands_S1)
        bands_S2_bc = spark.sparkContext.broadcast(prepper.bands_S2)
        size_bc = spark.sparkContext.broadcast(prepper.size)
        overlap_bc = spark.sparkContext.broadcast(prepper.overlap)

        #Download - parallel
        spark.sparkContext.parallelize(toRDD).repartition(partitions).foreach(lambda x: mapper_f(x, credentials_bc.value, rebuild_bc.value, bands_S1_bc.value, bands_S1_bc.value, size_bc.value, overlap_bc.value))

        logging.info("Preprocessing finished")

##        prepper.download()
    else:
        # Using docopt, this shouldn't be accessible
        # If the appropriate args aren't used, docopt will auto display help
        logging.warning(f"Shouldn't be able to reach this branch, due to docopt: args {args}")


if __name__ == "__main__":
    __main__()
