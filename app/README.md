# CYBELE Sentinel Preprocessing

Other documents:

- [Configuration Management](./docs/configuration-management.md)
- [Running via docker](./docs/running-docker.md)

## Installation

Install GDAL/OGR

``` 
sudo add-apt-repository ppa:ubuntugis/ppa && sudo apt-get update
sudo apt-get update
sudo apt-get install gdal-bin
sudo apt-get install libgdal-dev
export CPLUS_INCLUDE_PATH=/usr/include/gdal
export C_INCLUDE_PATH=/usr/include/gdal
pip install GDAL==$(gdal-config --version)
```

Install python library requirements using:

```
pip install -r requirements.txt 
```


## Credentials

Credentials should be a json file of the format:

``` json
{
    "username": "USERNAME",
    "password": "PASSWORD",
    "sentinel_url_endpoint": "https://scihub.copernicus.eu/dhus"
}
```


