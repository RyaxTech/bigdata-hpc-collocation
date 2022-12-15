#!/usr/bin/env python3
"""Run a sentinel download workflow."""
# std. lib
import json
import os
import sys
import math
from argparse import ArgumentParser
from pathlib import Path
import itertools
import zipfile

# external
import numpy as np
import rasterio as rio
from osgeo import gdal
from omegaconf import OmegaConf
from mpi4py import MPI
# local
import product_finder
import snapper
import senprep
import roiutil


SENTINEL_ROOT = "/var/satellite-data/"

def image_shape(filename):
    shape = rio.open(str(filename)).read(masked=False).shape
    print(filename, "has shape", shape)


def split(a, n):
    k, m = divmod(len(a), n)
    return list(a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def find_products(config, credentials, njobs=1):
    """Find products using a callback from `product_finder.SentinelProductFinder`."""
    finder = getattr(product_finder, config["callback_find_products"])
    product_list, other_find_results = finder(config, credentials)
    return product_list, other_find_results


def crop_image(s1_or_s2, filename, rebuild):
    # filename: <SENTINEL_ROOT>/<name>/ROI/S1/Collocated/S1_abc_S2_def.tif
    if not isinstance(filename, Path):
        filename = Path(filename)
    _, s1uuid, _, s2uuid = filename.stem.split(
        "_")  # e.g. s1_uuid = abc, s2_uuid = def
    cropdir = (
        filename.parent.parent / "Clipped"
    )  # go back to the /S1/ folder, as per above example
    roi_dir = filename.parent.parent.parent  # e.g. the ../ROI1/ part
    roi_no = roi_dir.name.replace("ROI", "")  # e.g. the '1' from ROI1
    roi_path = roi_dir / f"ROI{roi_no}.geojson"

    cropdir.mkdir(exist_ok=True)

    if s1_or_s2 == "S1":
        crop_filename = f"S1_roi{roi_no}_{s1uuid}.tif"
    else:
        crop_filename = f"S2_roi{roi_no}_{s2uuid}.tif"
    path_crop = cropdir / crop_filename
    if path_crop.exists() and not rebuild:
        print("CACHED CROP:", path_crop)
        return (s1_or_s2, path_crop, filename)

    gdal_result = gdal.Warp(
        str(path_crop),
        str(filename),
        cutlineDSName=str(roi_path),
        cropToCutline=True,
        dstNodata=999999999.0,
    )
    gdal_result = None
    gdal_result = gdal.Warp(
        str(path_crop),
        str(filename),
        cutlineDSName=str(roi_path),
        cropToCutline=True,
        dstNodata=999999999.0,
    )

    return (s1_or_s2, path_crop, filename)


def crop(to_crop):
    to_patch = []
    for s1_or_s2, fn_collocate in to_crop:
        to_patch.append(crop_image(s1_or_s2, fn_colloate, rebuild))
    return to_patch


def make_patches_from_image(s1_or_s2, filename_cropped, filename_collocated, config):
    # filename_cropped is like <SENTINEL_ROOT>/<name>/ROI<roi_no>/<S1_or_S2>/Clipped/<S1_or_S2>_roi<num>_uuid.tif
    patchdir = filename_cropped.parent / "PATCHES"
    patchdir.mkdir(exist_ok=True)

    # e.g. s1_uuid = abc, s2_uuid = def
    _, s1uuid, _, s2uuid = filename_collocated.stem.split("_")

    raster = rio.open(str(filename_cropped))
    raster_im = raster.read(masked=False)
    res = int(raster.res[0])
    gdal_dataset = gdal.Open(str(filename_cropped))

    height, width = config["size"]
    row_starts = np.arange(0, raster_im.shape[1], height)
    col_starts = np.arange(0, raster_im.shape[2], width)

    # basically, ...for r in rows: for c in cols... , but done outside a loop
    top_left_corners = itertools.product(col_starts, row_starts)

    patches = []
    for (col, row) in top_left_corners:
        patch_filename = (
            f"S1_{s1uuid}_S2_{s2uuid}_{row}_{col}_{width}x{height}.tif"
        )
        path_patch = str(patchdir / patch_filename)
        patches.append(path_patch)
        gdal.Translate(path_patch, gdal_dataset, format="GTiff", srcWin=[col, row, width, height])
    raster.close()
    # TODO finish
    return patches


def make_patches(to_patch):
    all_patches = []
    height, width = config["size"]
    for s1_or_s2, fn_cropped, fn_collocate in to_patch:
        all_patches.extend(make_patches_from_image(s1_or_s2, fn_cropped, fn_collocate))
    return all_patches


def snap_flow_mapper(product_set, snap_function, config, mount=None, rebuild=False):
    """Run a single set of products through the workflow.

    Inputs
    ------
    product_set : list of N-tuple
        Something like...
            [(s1), (s1_2)...]
            [(s1, s2), (s1_2, s2_2)...]
            [(s1_old, s1, s2), (s1_old_2, s1_2, s2_2)...]
    snap_function : function
        a method from the snapper module.
        should expect the same N-tuple format as provided by
        callback_find_products in the config
    mount : Path-like
        Prefix to append to paths
        Used if we mount `pwd` somewhere with docker and want
        to use simpler paths
    rebuild : bool
        Whether to force the SNAP graph processing tool to run
        already-completed products

    Outputs
    -------
    failures : list
        product ids and failure message, if an error occurred
    n_patches : int
        total number of patches generated
    """

    filenames_collocated = snap_function(product_set, config, mount, rebuild)

    # filenames_collocated is something like:
    # [("S1", filename_s1_collocated), ("S2", filename_s2_collocated)]
    all_patches = []
    for sat_type, filename in filenames_collocated:
        # image_shape(filename)
        s1_or_s2, filename_cropped, filename_collocated = crop_image(
            sat_type, filename, rebuild)
        # image_shape(filename_cropped)
        all_patches.extend(make_patches_from_image(
            s1_or_s2, filename_cropped, filename_collocated, config
        ))
    return all_patches


if __name__ == "__main__":
    parser = ArgumentParser("snap_flow")
    parser.add_argument("--config", required=True)
    parser.add_argument("--credentials", required=True)
    parser.add_argument("--mount", required=False)
    parser.add_argument("--njobs", default=1, required=False)
    parser.add_argument("--rebuild", required=False, default=False)
    parser.add_argument(
        "--output", help="Change output directory", required=False)
    args = parser.parse_args()

    config = args.config
    credentials = args.credentials

    # 'mount' is a helper arg when using docker, to specify where $PWD is mounted
    # to inside the docker image (e.g. -v $(pwd):/here/ suggests --mount "/here/")
    #
    # this lets us do '--config configurations/sample.json'
    # rather than     '--config /here/configurations/sample.json'
    if args.mount:
        config = os.path.join(args.mount, config)
        credentials = os.path.join(args.mount, credentials)

    if args.output:
        SENTINEL_ROOT = args.output
        senprep.SENTINEL_ROOT = args.output
        snapper.SENTINEL_ROOT = args.output

    # ========== Load and validate config
    config = OmegaConf.load(open(config))
    assert "callback_find_products" in config, "Need a callback to find product IDs"
    assert "callback_snap" in config, "Need a callback for running SNAP"
    assert "geojson" in config, "Need a geojson region of interest"

    print("GDAL CACHE MAX: ", gdal.GetCacheMax() / 1024 / 1024 / 1024, "GB")

    print("FINDER:", config.callback_find_products)
    print("SNAPPER:", config.callback_snap)

    # ========= Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    comm_size = comm.Get_size()

    # ========= Find products
    if rank == 0:
        product_sets, other_return_data = find_products(config, credentials)

    # ========= Distribute products to MPI processes
    if rank == 0:
        _product_sets = split(product_sets, comm_size)
    else:
        _product_sets = []
    _product_sets = comm.scatter(_product_sets, root=0)
    # ========= Run snap flow for each product
    # callback_snap defines the name of a function inside the snapper module
    # something like:
    #       f(product_tuple, config, mount, rebuild)

    # ======== This is now parallel
    # product_sets -> _product_sets
    # results -> _results
    snap_func = getattr(snapper, config.callback_snap)
    _results = []
    for p_set in _product_sets:
        _results.extend(snap_flow_mapper(p_set, snap_func, config, rebuild=args.rebuild))

    results = comm.gather(_results, root=0)
    if results:
        results = [r for result in results for r in result if result]
    if (rank == 0):
        print()
        # print(results)
        print("{} patches created from {} sets of products".format(len(results), len(product_sets)))
        # create zip...
        common_parent = os.path.commonpath(results)
        zipname = Path(common_parent) / "patches.zip"
        with zipfile.ZipFile(str(zipname), "w") as zf:
            for filename in results:
                zf.write(filename)
        print("Zip of patches created:", zipname)
