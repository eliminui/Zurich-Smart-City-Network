
import os
import json
import pandas as pd
import geopandas as gpd

print("Début préparation des données")

# -----------------------------
# Dossiers
# -----------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")
os.makedirs(data_dir, exist_ok=True)

# -----------------------------
# Fichiers source
# -----------------------------
csv1 = os.path.join(base_dir, "fichier_projet1.csv")
csv2 = os.path.join(base_dir, "fichier_projet7.csv")
json3 = os.path.join(base_dir, "projet2.json")
csv4 = os.path.join(base_dir, "projet3.csv")

csv5 = os.path.join(base_dir, "data_projects", "projet5", "projet5.csv")
csv6 = os.path.join(base_dir, "data_projects", "projet6", "projet6.csv")

gpkg8 = os.path.join(base_dir, "data_projects", "projet8", "projet8.gpkg")

# -----------------------------
# Fichiers de sortie
# -----------------------------
out1 = os.path.join(data_dir, "projet1.geojson")
out2 = os.path.join(data_dir, "projet7.geojson")
out3 = os.path.join(data_dir, "projet3.geojson")
out5 = os.path.join(data_dir, "projet5.geojson")
out6 = os.path.join(data_dir, "projet6.geojson")
out_laser = os.path.join(data_dir, "laser_scanners.geojson")
out_rtk = os.path.join(data_dir, "rtk_stations.geojson")
out_eth2 = os.path.join(data_dir, "eth2.geojson")
out_move_chill = os.path.join(data_dir, "move_chill_sensors.geojson")

# -----------------------------
# Bornes Zurich
# -----------------------------
ZURICH_LAT_MIN = 47.30
ZURICH_LAT_MAX = 47.45
ZURICH_LON_MIN = 8.45
ZURICH_LON_MAX = 8.65


def lv95_to_wgs84(x, y):
    y_aux = (x - 2600000.0) / 1000000.0
    x_aux = (y - 1200000.0) / 1000000.0

    lat = (
        16.9023892
        + 3.238272 * x_aux
        - 0.270978 * (y_aux ** 2)
        - 0.002528 * (x_aux ** 2)
        - 0.0447 * (y_aux ** 2) * x_aux
        - 0.0140 * (x_aux ** 3)
    )

    lon = (
        2.6779094
        + 4.728982 * y_aux
        + 0.791484 * y_aux * x_aux
        + 0.1306 * y_aux * (x_aux ** 2)
        - 0.0436 * (y_aux ** 3)
    )

    lat = lat * 100 / 36
    lon = lon * 100 / 36

    return lat, lon


def keep_only_zurich_points(df, lat_col="lat", lon_col="lon"):
    return df[
        (df[lat_col] >= ZURICH_LAT_MIN) &
        (df[lat_col] <= ZURICH_LAT_MAX) &
        (df[lon_col] >= ZURICH_LON_MIN) &
        (df[lon_col] <= ZURICH_LON_MAX)
    ].copy()


def point_feature(lat, lon, properties=None):
    if properties is None:
        properties = {}

    return {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [float(lon), float(lat)]
        },
        "properties": properties
    }


def polygon_feature(coordinates, properties=None):
    if properties is None:
        properties = {}

    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [coordinates]
        },
        "properties": properties
    }


def feature_collection(features):
    return {
        "type": "FeatureCollection",
        "features": features
    }


def save_geojson(output_path, data):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print("Créé :", output_path)


def build_point_features(df, lat_col, lon_col, properties_builder):
    features = []
    for row in df.itertuples(index=False):
        lat = getattr(row, lat_col)
        lon = getattr(row, lon_col)
        props = properties_builder(row)
        features.append(point_feature(lat, lon, props))
    return features


def build_projet5_hardbruecke_polygon(output_path):
    """
    Projet 5 :
    Le CSV ne contient pas de coordonnées exactes des capteurs/lignes.
    On représente donc la zone instrumentée de Hardbrücke.
    """
    hardbruecke_polygon = [
        [8.5188, 47.3862],
        [8.5212, 47.3862],
        [8.5212, 47.3848],
        [8.5188, 47.3848],
        [8.5188, 47.3862]
    ]

    feature = polygon_feature(
        coordinates=hardbruecke_polygon,
        properties={
            "name": "Pedestrian counting system",
            "layer": "Pedestrian counting system",
            "site": "VBZ-Haltestelle Hardbrücke"
        }
    )

    save_geojson(output_path, feature_collection([feature]))


def build_projet6_points(output_path):
    """
    Projet 6 :
    Utilise EKoord / NKoord (LV95), puis conversion en WGS84.
    On garde un point par station de mesure.
    """
    print("\n--- LECTURE projet6.csv ---")

    df6 = pd.read_csv(csv6)
    print("Colonnes disponibles :", list(df6.columns))
    print("Nombre de lignes :", len(df6))

    required_cols = ["EKoord", "NKoord"]
    for col in required_cols:
        if col not in df6.columns:
            raise ValueError(f"Colonne manquante dans projet6.csv : {col}")

    df6 = df6.dropna(subset=["EKoord", "NKoord"]).copy()

    coords = df6.apply(lambda r: lv95_to_wgs84(r["EKoord"], r["NKoord"]), axis=1)
    df6["lat"] = coords.str[0]
    df6["lon"] = coords.str[1]

    df6 = keep_only_zurich_points(df6, lat_col="lat", lon_col="lon")

    dedup_candidates = ["MSID", "ZSID", "lat", "lon"]
    dedup_cols = [c for c in dedup_candidates if c in df6.columns]
    if not dedup_cols:
        dedup_cols = ["lat", "lon"]

    df6 = df6.drop_duplicates(subset=dedup_cols).copy()

    def props_builder(row):
        props = {
            "name": "Motor vehicle traffic counting system",
            "layer": "Motor vehicle traffic counting system"
        }

        if hasattr(row, "ZSName") and pd.notna(getattr(row, "ZSName")):
            props["station_name"] = str(getattr(row, "ZSName"))

        if hasattr(row, "Achse") and pd.notna(getattr(row, "Achse")):
            props["street"] = str(getattr(row, "Achse"))

        if hasattr(row, "HNr") and pd.notna(getattr(row, "HNr")):
            props["house_number"] = str(getattr(row, "HNr"))

        if hasattr(row, "Richtung") and pd.notna(getattr(row, "Richtung")):
            props["direction"] = str(getattr(row, "Richtung"))

        return props

    features6 = build_point_features(
        df6,
        lat_col="lat",
        lon_col="lon",
        properties_builder=props_builder
    )

    save_geojson(output_path, feature_collection(features6))


# -----------------------------
# PROJET 1 : PIR + induction
# -----------------------------
df1 = pd.read_csv(csv1)
coords1 = df1.apply(lambda r: lv95_to_wgs84(r["OST"], r["NORD"]), axis=1)
df1["lat"] = coords1.str[0]
df1["lon"] = coords1.str[1]
df1 = keep_only_zurich_points(df1)

features1 = build_point_features(
    df1,
    lat_col="lat",
    lon_col="lon",
    properties_builder=lambda row: {
        "name": "Induction loops + PIR",
        "layer": "PIR + induction"
    }
)

save_geojson(out1, feature_collection(features1))


# -----------------------------
# PROJET 7 : Induction loops
# -----------------------------
df2 = pd.read_csv(csv2)
coords2 = df2.apply(lambda r: lv95_to_wgs84(r["E"], r["N"]), axis=1)
df2["lat"] = coords2.str[0]
df2["lon"] = coords2.str[1]
df2 = keep_only_zurich_points(df2)

features2 = build_point_features(
    df2,
    lat_col="lat",
    lon_col="lon",
    properties_builder=lambda row: {
        "name": "Induction loops",
        "layer": "Induction loops"
    }
)

save_geojson(out2, feature_collection(features2))


# -----------------------------
# PROJET 3 : Surveillance cameras
# -----------------------------
df4 = pd.read_csv(csv4)
df4 = df4.dropna(subset=["lat", "lon"])
df4 = keep_only_zurich_points(df4)

def build_camera_props(row):
    adresse = ""
    if hasattr(row, "adresse_beschreibung"):
        val = getattr(row, "adresse_beschreibung")
        adresse = "" if pd.isna(val) else str(val)

    return {
        "name": "Surveillance cameras",
        "adresse": adresse,
        "layer": "Surveillance cameras"
    }

features3 = build_point_features(
    df4,
    lat_col="lat",
    lon_col="lon",
    properties_builder=build_camera_props
)

save_geojson(out3, feature_collection(features3))


# -----------------------------
# LASER SCANNERS
# -----------------------------
with open(json3, "r", encoding="utf-8") as f:
    data3 = json.load(f)

save_geojson(out_laser, data3)


# -----------------------------
# RTK reference stations
# -----------------------------
rtk_feature = feature_collection([
    point_feature(
        47.4, 8.45,
        {
            "name": "RTK reference station (FP07)",
            "layer": "RTK reference stations"
        }
    ),
    point_feature(
        47.44, 8.48,
        {
            "name": "RTK reference station (BeLaAG01)",
            "layer": "RTK reference stations"
        }
    )
])

save_geojson(out_rtk, rtk_feature)


# -----------------------------
# ETH2 : station géodésique
# -----------------------------
eth2_feature = feature_collection([
    point_feature(
        47.40717, 8.51061,
        {
            "name": "Permanent GNSS reference station (ETH2)",
            "layer": "Permanent GNSS reference station"
        }
    )
])

save_geojson(out_eth2, eth2_feature)


# -----------------------------
# PROJET 5 : Pedestrian counting system
# -----------------------------
build_projet5_hardbruecke_polygon(out5)


# -----------------------------
# PROJET 6 : Motor vehicle traffic counting system
# -----------------------------
build_projet6_points(out6)


# -----------------------------
# PROJET 8 : Move and Chill sensors
# -----------------------------
print("\n--- LECTURE projet8.gpkg ---")

gdf8 = gpd.read_file(gpkg8)

print("Colonnes disponibles :", list(gdf8.columns))
print("Nombre de lignes :", len(gdf8))

sensor_col = "sensor_eui"
lat_col = "latitude"
lon_col = "longitude"

gdf8 = gdf8.dropna(subset=[sensor_col, lat_col, lon_col]).copy()

df8_mean = (
    gdf8.groupby(sensor_col, as_index=False)[[lat_col, lon_col]]
    .mean()
)

df8_mean = keep_only_zurich_points(df8_mean, lat_col=lat_col, lon_col=lon_col)

features8 = build_point_features(
    df8_mean,
    lat_col=lat_col,
    lon_col=lon_col,
    properties_builder=lambda row: {
        "name": f"Smart IoT sensor node (SENSOR_EUI = {getattr(row, sensor_col)})",
        "sensor_eui": str(getattr(row, sensor_col)),
        "layer": "Move and Chill sensors"
    }
)

save_geojson(out_move_chill, feature_collection(features8))

print("Préparation terminée.")