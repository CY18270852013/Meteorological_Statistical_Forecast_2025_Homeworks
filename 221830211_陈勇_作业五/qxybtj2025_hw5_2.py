import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from eofs.standard import Eof
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.ticker as mticker

# 读取数据并调整经度坐标
ds = xr.open_dataset(r'C:\Users\Chen Yong\Desktop\气象统计预报上机作业\221830211_陈勇_作业五\HadISST_sst.nc')
ds = ds.assign_coords(longitude=(ds.longitude % 360)).sortby('longitude')

# 异常值处理，SST范围：-2°C 到 35°C，超出范围设为NaN
sst_clean = ds['sst'].where((ds.sst >= -2) & (ds.sst <= 35))

# 选择北太平洋区域
region = sst_clean.sel(
    latitude=slice(60, 20),  # 纬度降序排列
    longitude=slice(120, 240),
    time=slice('1900-01-01', '2020-12-31')
)

# 计算年度异常
sst_annual = region.resample(time='Y').mean('time', skipna=False)
sst_clim = sst_annual.mean(dim='time', skipna=True)
ssta = sst_annual - sst_clim

# 处理缺失值
ssta = ssta.where(np.isfinite(ssta), 0)

# 纬度加权与区域平均
coslat = np.cos(np.deg2rad(ssta.latitude))
weights = np.sqrt(coslat).values[:, np.newaxis]
weights = weights * np.ones((1, ssta.longitude.size))

# 计算加权区域平均
weighted_mean = (ssta * weights).sum(dim=['latitude', 'longitude']) / (weights.sum() * ssta.longitude.size)
ssta_adjusted = ssta - weighted_mean

# EOF分析
stacked = ssta_adjusted.stack(space=('latitude', 'longitude'))
data = stacked.transpose('time', 'space').values

solver = Eof(data, weights=weights.ravel())
eof1 = solver.eofsAsCorrelation(neofs=2)[1]
eof_2d = eof1.reshape(ssta.latitude.size, ssta.longitude.size)
pc1 = solver.pcs(npcs=2, pcscaling=1)[:, 1]
variance = solver.varianceFraction(neigs=2)[1] * 100

# 绘图设置
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})

# 组合图形绘制
fig = plt.figure(figsize=(14, 10))

# 空间模态
ax1 = fig.add_axes([0.08, 0.37, 0.84, 0.6], projection=ccrs.PlateCarree(central_longitude=180))
ax1.set_extent([120, 240, 20, 60], crs=ccrs.PlateCarree())

def format_lon(lon):
    if lon > 180:
        return f"{int(360-lon)}°W"
    else:
        return f"{int(lon)}°E"

cf = ax1.contourf(ssta.longitude, ssta.latitude, eof_2d,
                 levels=np.linspace(-1, 1, 11),
                 cmap='RdBu_r',
                 transform=ccrs.PlateCarree(),
                 extend='both')

ax1.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='k', zorder=2)
ax1.coastlines(resolution='50m', linewidth=0.5, zorder=2)
ax1.set_xticks(np.arange(120, 241, 20), crs=ccrs.PlateCarree())
ax1.set_xticklabels([format_lon(lon) for lon in np.arange(120, 241, 20)])
ax1.set_yticks(np.arange(20, 61, 10))
ax1.yaxis.set_major_formatter(LatitudeFormatter())
ax1.set_title(f'EOF2 (Variance Explained: {variance:.1f}%)', pad=20)
gl = ax1.gridlines(crs=ccrs.PlateCarree(),
                  draw_labels=False,
                  linewidth=0.5,
                  color='gray',
                  alpha=0.5,
                  linestyle='--')
gl.xlocator = mticker.FixedLocator(np.arange(120, 241, 10))  # 每10度经线
gl.ylocator = mticker.FixedLocator(np.arange(20, 61, 5))    # 每5度纬线

# 颜色条
cax = fig.add_axes([0.2, 0.42, 0.6, 0.02])
cbar = plt.colorbar(cf, cax=cax, orientation='horizontal')
cbar.set_ticks(np.arange(-1, 1.1, 0.25))
cbar.set_label('Correlation Coefficient', labelpad=8)

# 时间序列子图
ax2 = fig.add_axes([0.08, 0.08, 0.84, 0.2])

years = ssta_adjusted.time.dt.year.values
bar_colors = np.where(pc1 > 0, '#d62728', '#1f77b4')

ax2.bar(years, pc1,
       width=0.8,
       color=bar_colors,
       edgecolor='none')

ax2.set_xlim(1899.5, 2020.5)
ax2.set_ylim(-3, 3)
ax2.set_xticks(np.arange(1900, 2021, 10))
ax2.set_yticks(np.arange(-3, 4, 1))
ax2.grid(axis='y', linestyle=':', alpha=0.3)

ax2.set_title('PC2 Time Series', pad=10)
ax2.set_xlabel('Year', labelpad=8)
ax2.set_ylabel('Normalized Units', labelpad=8)

plt.savefig('PDO_analysis.png', dpi=300, bbox_inches='tight')
plt.show()