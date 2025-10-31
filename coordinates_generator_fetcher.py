# requirements: geopandas, shapely, fiona, pyproj, requests (for images)
import geopandas as gpd
from shapely.geometry import Point, box
import random
import requests
import os
import pandas as pd


# 1) Load region polygons (shapefile or geojson you downloaded)
regions_gdf = gpd.read_file("ga_eco_l3.shp")  # change filename
print(regions_gdf.head())

# 2) (Optional) restrict to the particular set of region names you want
# regions_gdf = regions_gdf[regions_gdf['US_L3NAME'].isin(['Blue Ridge','Piedmont',...])]

# 3) Create a single polygon for the whole state
state_poly = regions_gdf.unary_union

# Helper: sample a point inside the state polygon
def sample_point_in_polygon(polygon, bbox=None, max_tries=1000):
    if bbox is None:
        minx, miny, maxx, maxy = polygon.bounds
    else:
        minx, miny, maxx, maxy = bbox
    for _ in range(max_tries):
        x = random.uniform(minx, maxx)
        y = random.uniform(miny, maxy)
        p = Point(x, y)
        if polygon.contains(p):
            return p
    raise RuntimeError("Failed to sample point inside polygon")

# 4) Generate labeled points (example: sample M points total)
M = 2000
pts = []
for i in range(M):
    p = sample_point_in_polygon(state_poly)
    pts.append(p)

pts_gdf = gpd.GeoDataFrame(geometry=pts, crs=regions_gdf.crs)

# Spatial join to assign regions
joined = gpd.sjoin(pts_gdf, regions_gdf, how='left', predicate='within')
joined = joined.rename(columns={'US_L3NAME': 'region'})
print(joined[['geometry','region']].head())

# Convert to lat/lon
joined = joined.to_crs(epsg=4326)
joined['lon'] = joined.geometry.x
joined['lat'] = joined.geometry.y

# -----------------------------
# 5) Fetch satellite images and map to regions
# -----------------------------
API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")  # replace with your Google Maps API key
SAVE_DIR = "images"
os.makedirs(SAVE_DIR, exist_ok=True)

def fetch_satellite_image(lat, lon, region_label, idx, zoom=16, size="640x640"):
    base_url = "https://maps.googleapis.com/maps/api/staticmap"
    url = f"{base_url}?center={lat},{lon}&zoom={zoom}&size={size}&maptype=satellite&key={API_KEY}"
    try:
        r = requests.get(url)
        if r.status_code == 200:
            # Save with a filename containing the index and region
            filename = f"{SAVE_DIR}/{idx}_{region_label.replace(' ','_')}.png"
            with open(filename, "wb") as f:
                f.write(r.content)
            return filename
        else:
            print(f"Failed to fetch image for {lat},{lon}: HTTP {r.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching image for {lat},{lon}: {e}")
        return None

# Loop over all coordinates and fetch images
image_paths = []
for idx, row in joined.iterrows():
    if pd.isna(row['region']):
        continue  # skip if no region assigned
    img_path = fetch_satellite_image(row['lat'], row['lon'], row['region'], idx)
    if img_path:
        image_paths.append((img_path, row['region']))

print(f"Downloaded {len(image_paths)} images out of {len(joined)} coordinates.")
