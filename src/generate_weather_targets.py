import os
import json
import random

from datetime import datetime, timedelta

import numpy as np
import geopandas as gpd


def mollweide_to_lat_lon(x, y):
    R = 6378e3
    theta = np.arcsin(y / R / np.sqrt(2))
    phi = np.arcsin((2*theta + np.sin(2 * theta)) / np.pi)
    lambd = (np.pi * x / (2 * R * 2**0.5 * np.cos(theta)))
    return phi * 180 / np.pi, lambd * 180 / np.pi


def random_datetime():
    day_offset = random.randint(0, 364)
    afternoon = random.randint(0, 1) == 1
    return datetime(2022, 1, 1, 18 if afternoon else 6) + timedelta(days=day_offset)


if __name__ == "__main__":
    if not os.path.isfile("data/GHS_UCDB_GLOBE_R2024A.gpkg"):
        assert False, f"Please download GHS-UCDB R2024A as 'GHS_UCDB_GLOBE_R2024A.gpkg' and place it in the data folder."
    data = gpd.read_file("data/GHS_UCDB_GLOBE_R2024A.gpkg", layer="GHS_UCDB_THEME_GHSL_GLOBE_R2024A")
    centroids = gpd.read_file("data/GHS_UCDB_GLOBE_R2024A.gpkg", layer="UC_centroids")

    top_urban_centers = []
    for row in data.sort_values(by="GH_POP_TOT_2025", ascending=False).iloc[:1000].iterrows():
        row = row[1].to_dict()
        id_uc = row["ID_UC_G0"]
        x, y = centroids.loc[centroids["ID_UC_G0"] == id_uc, ("PWCentroidX", "PWCentroidY")].values[0]
        lat, lon = mollweide_to_lat_lon(x, y)
        name = row["GC_UCN_MAI_2025"]
        population = row["GH_POP_TOT_2025"]
        top_urban_centers.append(dict(
            name=name, population=population,
            latitude=lat.item(), longitude=lon.item(),
            )
        )

    with open("data/top_urban_centers.json", "w", encoding="utf-8") as f:
        json.dump(top_urban_centers, f)

    pairs = []
    for _ in range(100):
        pairs.append(dict(datetime=random_datetime(), location=random.choice(top_urban_centers)))

    for i, pair in enumerate(pairs):
        pair["datetime"] = pair["datetime"].strftime("%Y-%m-%dT%H:%M:%S")
        pairs[i] = pair

    with open("data/weather_evaluation_targets.json", "w") as f:
        json.dump(pairs, f)