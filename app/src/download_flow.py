#!/usr/bin/env python3
"""Run a sentinel download workflow."""
import json
import os
import sys
from pathlib import Path

from metaflow import FlowSpec, step, Parameter
import product_finder
import senprep


SENTINEL_ROOT = "/var/satellite-data/"


class SentinelDownload(FlowSpec):
    """SenPrep v2. Implemented through callbacks."""

    config = Parameter("config", help="Configuration json", required=True)
    credentials_file = Parameter(
        "credentials", help="SentinelSat Credentials", required=True
    )
    credentials_file_earthdata = Parameter(
        "credentials_earthdata", help="Earthdata Credentials"
    )
    mount = Parameter("mount", help="Where the current dir is mounted", required=False)

    @step
    def start(self):
        """Read and validate configuration.

        This step ensures that the configuration has the following fields:
        - dates
        - callback_find_products
        - geojson
        """
        # Need to wrap the metaflow parameters to modify with the mount path
        # since we're potentially/usually running from within Docker
        if self.mount:
            self.cfg = json.load(open(os.path.join(self.mount, self.config), "r"))
            self.credentials = os.path.join(self.mount, self.credentials_file)
            self.credentials_ed = os.path.join(
                self.mount, self.credentials_file_earthdata
            )
        else:
            self.cfg = json.load(open(self.config, "r"))
            self.credentials = self.credentials_file
            self.credentials_ed = self.credentials_file
        assert (
            "dates" in self.cfg
        ), "Need to include (yyyymmdd, yyyymmdd) start and end dates."
        assert (
            "callback_find_products" in self.cfg
        ), "Need a SentinelProductFinder callback."
        assert "geojson" in self.cfg, "Need a geojson region of interest."
        print("FINDER", self.cfg["callback_find_products"])
        self.next(self.find_products)

    @step
    def find_products(self):
        """Find products using a callback from `product_finder.SentinelProductFinder`.

        The callback is ran to generate a list of ids by querying and filtering the sentinelsat api.

        For download, all we need is for the product_finder to return a dict something like
        {'ids': [(id1, id2), (id3, id4)]}, which will be flattened to download each of
        [id1, id2, id3, id4]
        """
        finder = getattr(product_finder, self.cfg["callback_find_products"])
        self.product_list, self.other_find_results = finder(self.cfg, self.credentials)
        self.products = [
            product
            for product_tuple in self.product_list
            for product in product_tuple["ids"]
        ]
        self.next(self.download)

    @step
    def download(self):
        """ForEach found product, download."""
        self.downloaded = []
        api = senprep.load_api(self.credentials)
        earthdata_auth = None
        if self.credentials_ed and Path(self.credentials_ed).exists():
            earthdata_auth = json.load(open(self.credentials_ed))
        for i, product in enumerate(self.products):
            print(
                "DL {i}/{n}: {uuid}".format(
                    i=i + 1,
                    n=len(self.products),
                    uuid=product.uuid,
                ),
                end="",
            )
            metadata = api.get_product_odata(product.uuid)
            s1_or_s2 = metadata["title"][:2].lower()
            if metadata["Online"] == True:
                print(" - (online -> SentinelSat)")
                api.download(product.uuid, directory_path=SENTINEL_ROOT, checksum=True)
            else:
                if s1_or_s2 == "s2":
                    print(" - (offline S2 -> GCS)")
                    senprep.download_S2_GCS(product)
                elif s1_or_s2 == "s1":
                    print(" - (offline S1 -> NOAA)")
                    if not earthdata_auth:
                        print("NO EARTHDATA CREDENTIALS. FAIL.")
                    else:
                        senprep.download_S1_NOAA(product, auth=earthdata_auth)
                else:
                    raise "Invalid odata. No alternate downloader for offline product."
            self.downloaded.append(product.uuid)
        self.next(self.end)

    @step
    def end(self):
        """Display to user what has been downloaded."""
        for product in self.downloaded:
            print("DOWNLOADED {}".format(product))
        return


if __name__ == "__main__":
    SentinelDownload()
