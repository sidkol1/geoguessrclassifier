# requirements: geopandas, shapely, fiona, pyproj, requests (for images)
import geopandas as gpd
from shapely.geometry import Point, box
import random
import requests
import os

# 1) Load region polygons (shapefile or geojson you downloaded)
regions_gdf = gpd.read_file("ga_eco_l3.shp")  # change filename
# inspect: print(regions_gdf.columns); find the region-name column (e.g., 'ECO_NAME' or 'LEVEL3')
print(regions_gdf.head())

# 2) (Optional) restrict to the particular set of region names you want, or re-label.
# regions_gdf = regions_gdf[regions_gdf['ECO_NAME'].isin(['Blue Ridge','Piedmont',...])]

# 3) Create a single polygon for the whole state (union of all parts)
state_poly = regions_gdf.unary_union

# helper: sample a point inside the state polygon by rejection sampling
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

# 4) generate labeled points (example: sample M points total)
M = 2000
pts = []
labels = []
for i in range(M):
    p = sample_point_in_polygon(state_poly)
    pts.append(p)

pts_gdf = gpd.GeoDataFrame(geometry=pts, crs=regions_gdf.crs)

joined = gpd.sjoin(pts_gdf, regions_gdf, how='left', predicate='within')
joined = joined.rename(columns={'US_L3NAME': 'region'})  # use the right column
print(joined[['geometry','region']].head())

joined = joined.to_crs(epsg=4326)
joined['lon'] = joined.geometry.x
joined['lat'] = joined.geometry.y

