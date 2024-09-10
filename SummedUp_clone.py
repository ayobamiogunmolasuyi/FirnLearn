




# For opening, viewing, and printing dataset info
import numpy as np
import pandas as pd

# For plotting data and metadata
import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path as mpath
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
matplotlib_inline.backend_inline.set_matplotlib_formats('png')

# Create a dataframe of info for each profile
dfs_info = []
for i in df.Profile.unique():

    # Create dataframe for each profile
    dfi = df[df.Profile==i].reset_index(drop=True)

    # Get profile information
    idx = np.array(dfi.Profile)[0]
    cit = np.array(dfi.Citation)[0]
    method = np.array(dfi.Method)[0]
    timestamp = np.array(dfi.Timestamp)[0]
    lat = np.array(dfi.Latitude)[0]
    lon = np.array(dfi.Longitude)[0]
    elev = np.array(dfi.Elevation)[0]
    
    # Get max depth
    if dfi.Stop_Depth[0] != -9999:
        max_depth = np.array(dfi.Stop_Depth)[-1]
    else:
        max_depth = np.array(dfi.Midpoint)[-1]
            
    df_info = pd.DataFrame(data={'Profile':idx,'Citation':cit,'Method':method,'Timestamp':timestamp,
                                 'Latitude':lat,'Longitude':lon,'Elevation':elev,'Max_Depth':max_depth},
                           index=[0])
    dfs_info.append(df_info)

# Concat individual profiles into a single dataframe
df_meta = pd.concat(dfs_info).reset_index(drop=True)

# Show first few rows
df_meta.head()

# Split dataset
df_greenland = df[df.Latitude>0]
df_antarctica = df[df.Latitude<0]

# Split metadata
df_greenland_meta = df_meta[df_meta.Latitude>0]
df_antarctica_meta = df_meta[df_meta.Latitude<0]

# Print info by ice sheet
print('Antarctica')
print('Number of profiles: {}'.format(len(df_antarctica_meta)))
print('Max core depth: {} m'.format(max(df_antarctica_meta.Max_Depth)))
print('Earliest measurement: {}'.format(df_antarctica_meta.Timestamp.min()))
print('Most recent measurement: {}'.format(df_antarctica_meta.Timestamp.max()))

print('\nGreenland')
print('Number of profiles: {}'.format(len(df_greenland_meta)))
print('Max core depth: {} m'.format(max(df_greenland_meta.Max_Depth)))
print('Earliest measurement: {}'.format(df_greenland_meta.Timestamp.min()))
print('Most recent measurement: {}'.format(df_greenland_meta.Timestamp.max()))

# Set up figure for plotting 
fig = plt.figure(figsize=(5,5))

# Create axis with correct projection and extent
ax = plt.subplot(1,1,1,projection=ccrs.SouthPolarStereo())
ax.set_extent([-180, 180, -90, -65], ccrs.PlateCarree())
ax.axis('off')

# Add features
glacier = cfeature.NaturalEarthFeature('physical','glaciated_areas','50m',
                                       edgecolor='none',facecolor='w')
iceshelves = cfeature.NaturalEarthFeature('physical','antarctic_ice_shelves_polys','50m',
                                          edgecolor='none',facecolor='whitesmoke')
ax.add_feature(glacier,zorder=0)
ax.add_feature(iceshelves,zorder=1)
ax.coastlines(color='gray',zorder=2)

# Plot locations
ax.scatter(df_antarctica_meta.Longitude,df_antarctica_meta.Latitude,
           transform=ccrs.PlateCarree(),zorder=3,
           c='black',s=50,edgecolor='black',lw=0.5)

plt.tight_layout()

df_antarctica.to_csv("df_antarctica.csv")

