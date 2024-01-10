#!/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate XCSoar's repository file (https://download.xcsoar.org/repository).

Execute in the git repository root dir.
"""

import datetime
import json
from pathlib import Path
import subprocess
import sys
import requests
import re

from iso3166 import countries


def git_commit_datetime(filename: Path) -> datetime.datetime:
    """Return naive UTC datetime of filename's last git commit."""
    cmd = ["git", "log", "-1", "--format=%ct", "--", filename]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    out, _ = p.communicate()
    try:
        NewDate = datetime.datetime(out)
        return datetime.datetime.utcfromtimestamp(int(out))
    except:
        return datetime.datetime.now()


def guess_area(name: str) -> str:
    """From name (e.g. USA_PG_REG1-4), try to guess and return the ISO3166.1-alpha2 code, else empty string."""
    area = ""
    prefix = name.split(".")[0].split("_")[0]
    try:
        area = countries.get(prefix).alpha2.lower()
    except KeyError:
        print(f"Could not guess the country code (ISO 3166 alpha2) for: {name}")
    return area


def generate_content(data_dir: Path, url: str) -> str:
    """data_dir/$TYPE/[country,region,global]/*.*"""
    rv = ""
    for xcs_type in sorted(data_dir.iterdir()):
        for geo in sorted(xcs_type.iterdir()):
            rv += f"\n# Data location: {data_dir.name}, type: {xcs_type.name}, geography: {geo.name}.\n"
            for datafile in sorted(geo.iterdir()):
                rv += f"""
name={datafile.name}
uri={url + str(datafile.relative_to(data_dir))}
type={xcs_type.name}
area={guess_area(datafile.stem)}
update={git_commit_datetime(datafile).date().isoformat()}
"""
    return rv


def generate_source(data_dir: Path, url: str) -> str:
    """data_dir/$TYPE/[country,region,global]/*.*"""
    rv = ""
    for xcs_type in sorted(data_dir.iterdir()):
        for geo in sorted(xcs_type.iterdir()):
            rv += f"\n# Data location: {data_dir.name}, type: {xcs_type.name}, geography: {geo.name}.\n"
            for datafile in sorted(geo.iterdir()):
                rv += f"""
name={str(datafile.stem) + "_HighRes.xcm"}
uri={url + str(datafile.relative_to(data_dir)).replace(".json","_HighRes.xcm" )}
type={xcs_type.name}
area={guess_area(datafile.stem)}
update={git_commit_datetime(datafile).date().isoformat()}

name={str(datafile.stem) + ".xcm"}
uri={url + str(datafile.relative_to(data_dir)).replace(".json",".xcm" )}
type={xcs_type.name}
area={guess_area(datafile.stem)}
update={git_commit_datetime(datafile).date().isoformat()}
"""
    return rv


def generate_asp_openaip():
    """Generate openaip repository entries from cloud storage (airspace)"""
    base_url = "https://storage.googleapis.com/29f98e10-a489-4c82-ae5e-489dbcd4912f/"
    url = base_url
    openaip_index = ""
    rv = ""

    while True:
        response = requests.get(url)
        xml_data = response.text
        openaip_index += response.text

        # Search for the NextMarker tag in the XML data
        next_marker = None
        match = re.search(r"<NextMarker>(.*?)</NextMarker>", xml_data)
        if match:
            next_marker = match.group(1)
        if next_marker is None:
            break

        url = f"{base_url}?marker={next_marker}"

    contents = re.findall(r"<Contents>(.*?)</Contents>", openaip_index)
    for content in contents:
        key = re.search(r"<Key>(.*?)</Key>", content)
        if key.group(1).__contains__("asp_extended.txt"):
            size = re.search(r"<Size>(.*)</Size>", content)
            size = int(size.group(1))
            if size >= 384:
                print("OK: " + key.group(1) + " " + str(size))
                updatedate = re.search(r"<LastModified>(.*?)</LastModified>", content)
                countrycode = str.upper(str(key.group(1)[0:2]))
                countryname = countries.get(countrycode).name
                rv += f"""
name={countrycode + "-ASP-National" + "-OpenAIP.txt"}
uri={base_url + key.group(1)}
type=airspace
description={countryname + " Airspace from OpenAIP"}
area={countrycode}
update={updatedate.group(1)}
"""
    return rv


def json_uri(json_filename: Path) -> str:
    """Return the value of json_filename's "uri" key."""
    data = json.load(json_filename.open())
    return data["uri"]


def json_description(json_filename: Path) -> str:
    """Return the value of json_filename's "description" key."""
    data = json.load(json_filename.open())
    if "description" in data:
        return data["description"]


def generate_remote(data_dir: Path) -> str:
    """data_dir/$TYPE/[country,region,global]/*.*"""
    rv = ""
    for xcs_type in sorted(data_dir.iterdir()):
        for geo in sorted(xcs_type.iterdir()):
            rv += f"\n# Data location: {data_dir.name}, type: {xcs_type.name}, geography: {geo.name}.\n"
            for datafile in sorted(geo.iterdir()):
                rv += f"""
name={datafile.stem}
uri={json_uri(datafile)}
description={json_description(datafile)}
type={xcs_type.name}
area={guess_area(datafile.stem)}
update={git_commit_datetime(datafile).date().isoformat()}
"""
                if json_description(datafile) != None:
                    rv += f"""description={json_description(datafile)}
"""
    return rv


if __name__ == "__main__":
    """
    ./data/[content,remote,source]/$TYPE/[country,region,global]/*.*
    """
    root_dir = Path("data")
    content_dir = root_dir / Path("content")
    source_dir = root_dir / Path("source")
    remote_dir = root_dir / Path("remote")

    base_url = "https://download.xcsoar.org/"

    repo = ""
    repo += generate_content(data_dir=content_dir, url=base_url + "content/")
    repo += generate_source(data_dir=source_dir, url=base_url + "source/")
    repo += generate_remote(data_dir=remote_dir)
    repo += generate_asp_openaip()

    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "repository"

    with open(out_path, "w") as f:
        f.write(repo)
    print(f"Created: {out_path}")
