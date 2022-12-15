#!/usr/bin/env python
"""Run a sentinel download workflow."""
import json
import os
import sys
from pathlib import Path
from argparse import ArgumentParser
from functools import partial


#from metaflow import FlowSpec, step, Parameter
from omegaconf import OmegaConf
import product_finder
import senprep
import pyspark

SENTINEL_ROOT = "/var/satellite-data/"


def find_products(config, credentials):
    """Find products using a callback from `product_finder.SentinelProductFinder`.

    The callback is ran to generate a list of ids by querying and filtering the sentinelsat api.

    For download, all we need is for the product_finder to return a dict something like
    {'ids': [(id1, id2), (id3, id4)]}, which will be flattened to download each of
    [id1, id2, id3, id4]
    """
    finder = getattr(product_finder, config["callback_find_products"])
    product_list, other_find_results = finder(config, credentials)
    products = [
        product for product_tuple in product_list for product in product_tuple["ids"]
    ]
    return products


def download_product_new(product, credentials, credentials_gs, earthdata_auth, alternates):
    api = senprep.load_api(credentials)
    metadata = api.get_product_odata(product.uuid)
    print(metadata)
    s1_or_s2 = metadata["title"][:2].lower()

    print(f"{product.uuid}", end="")
    fn_safe = Path(SENTINEL_ROOT) / (product.title + ".SAFE")
    fn_zip = Path(SENTINEL_ROOT) / (product.title + ".zip")
    if fn_safe.exists() or fn_zip.exists():
        print("already downloaded")
        return product.uuid

    if metadata["Online"] == True and False:
        print(" - (online -> SentinelSat)")
        api.download(product.uuid, directory_path=SENTINEL_ROOT, checksum=True)
    elif alternates:
        if s1_or_s2 == "s2":
            print(" - (offline S2 -> GCS)")
            _credentials_gs = pyspark.SparkFiles.get("credentials_gs.json")
            senprep.download_S2_GCS_py(product, _credentials_gs)
        elif s1_or_s2 == "s1":
            print(" - (offline S1 -> NOAA)")
            if earthdata_auth:
                senprep.download_S1_NOAA_py(product, auth=earthdata_auth)
            else:
                print("Couldn't download offline S1 - no earthdata login")
        else:
            raise "Invalid odata. No alternate downloader for offline product."
    else:
        print("Products offline and no alternate downloader specified.")

    return product.uuid


def download_product(product, credentials, earthdata_auth, alternates, api):
    metadata = api.get_product_odata(product.uuid)
    s1_or_s2 = metadata["title"][:2].lower()

    print(f"{product.uuid}", end="")

    if metadata["Online"] == True:
        print(" - (online -> SentinelSat)")
        api.download(product.uuid, directory_path=SENTINEL_ROOT, checksum=True)
    elif alternates:
        if s1_or_s2 == "s2":
            print(" - (offline S2 -> GCS)")
            senprep.download_S2_GCS_py(product)
        elif s1_or_s2 == "s1":
            print(" - (offline S1 -> NOAA)")
            if earthdata_auth:
                senprep.download_S1_NOAA(product, auth=earthdata_auth)
            else:
                print("Couldn't download offline S1 - no earthdata login")
        else:
            raise "Invalid odata. No alternate downloader for offline product."
    else:
        print("Products offline and no alternate downloader specified.")

    return product.uuid


def download(
    products,
    credentials=None,
    credentials_ed=None,
    alternates=True,
):
    """ForEach found product, download."""
    downloaded = []
    api = senprep.load_api(credentials)
    earthdata_auth = None
    if credentials_ed:
        earthdata_auth = json.load(open(credentials_ed))
    for i, product in enumerate(products):
        print(f"DL {i+1}/{len(products)}: ", end="")
        download_product(
            product, credentials, earthdata_auth, alternates=alternates, api=api
        )
        downloaded.append(product.uuid)
    return downloaded


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--credentials", help="Credentials for SentinelSat", required=True
    )
    parser.add_argument(
        "--credentials_ed", help="Credentials for Earthdata", required=False
    )
    parser.add_argument(
        "--credentials_gs", help="Credentials for Sentinel-2 in Google Storage", required=False)
    parser.add_argument("--mount", required=False)
    parser.add_argument(
        "--alternate-downloaders",
        help="Use alternate download sources if needed",
        required=False,
    )
    parser.add_argument(
        "--output", help="Change output directory", required=False)
    args = parser.parse_args()

    config = args.config
    credentials = args.credentials
    credentials_ed = args.credentials_ed
    credentials_gs = args.credentials_gs
    # 'mount' is a helper arg when using docker, to specify where $PWD is mounted
    # to inside the docker image (e.g. -v $(pwd):/here/ suggests --mount "/here/")
    #
    # this lets us do '--config configurations/sample.json'
    # rather than     '--config /here/configurations/sample.json'
    if args.mount:
        config = os.path.join(args.mount, config)
        credentials = os.path.join(args.mount, credentials)
        credentials_ed = os.path.join(args.mount, credentials_ed)
        credentials_gs = os.path.join(args.mount, credentials_gs)
    if args.output:
        SENTINEL_ROOT = args.output
        senprep.SENTINEL_ROOT = args.output

    print(config)
    # ========== Load and validate config
    config = OmegaConf.load(open(config))
    assert "dates" in config, "Need start and end dates ((yyyymmdd, yyyymmdd))"
    assert "callback_find_products" in config, "Need a callback to find product IDs"
    assert "geojson" in config, "Need a geojson region of interest"
    print("FINDER:", config.callback_find_products)

    api = senprep.load_api(credentials)
    earthdata_auth = json.load(open(credentials_ed))

    # ========== Find products
    print("Finding products")
    product_ids = find_products(config, credentials)
    print("{} products".format(len(product_ids)))

    def check_product_exists(product):
        fn_safe = Path(SENTINEL_ROOT) / (product.title + ".SAFE")
        fn_zip = Path(SENTINEL_ROOT) / (product.title + ".zip")
        fn_dir = Path(SENTINEL_ROOT) / (product.title)
        return fn_safe.exists() or fn_zip.exists() or fn_dir.exists()

    existing, non_existing = [], []
    for p in product_ids:
        if check_product_exists(p):
            existing.append(p)
        else:
            non_existing.append(p)
    print("{} exist. {} need downloaded.".format(len(existing), len(non_existing)))

    # ========== Initialize spark session
    sc = pyspark.SparkContext()
    sc.addFile("demo4_preprocessing/src/senprep.py")
    sc.addFile("demo4_preprocessing/src/configutil.py")
    sc.addFile("demo4_preprocessing/src/roiutil.py")
    sc.addFile("demo4_preprocessing/src/sen_plot.py")
    sc.addFile("demo4_preprocessing/credentials.json")
    sc.addFile("demo4_preprocessing/credentials_cd.json")
    sc.addFile("demo4_preprocessing/credentials_ed.json")
    sc.addFile("demo4_preprocessing/credentials_gs.json")
    import senprep

    # ========== Download each product
    download_mapper = partial(
        download_product_new,
        credentials=credentials,
        credentials_gs=credentials_gs,
        earthdata_auth=earthdata_auth,
        alternates=True,
    )
    # FIXME: configure the number of partitions to be roughly 2x the number of cores
    print("Mapping product ids")
    product_ids_rdd = sc.parallelize(non_existing, 4)
    results_rdd = product_ids_rdd.map(lambda x: download_mapper(x))
    for result in results_rdd.collect():
        print("DOWNLOADED {}".format(result))
