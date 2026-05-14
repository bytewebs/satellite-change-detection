import geopandas as gpd

# Read shapefile
gdf = gpd.read_file("FullAOI_Export.shp")

# Convert first geometry to GeoJSON
geojson_geom = gdf.geometry.iloc[0].__geo_interface__

print(geojson_geom)