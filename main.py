import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import math
from geopy.distance import geodesic

# === Load Data ===
df = pd.read_excel("dataset.xlsx")
gdf = gpd.read_file("penelitian.geojson")

# === Normalize & Clean Names ===
df['kecamatan_clean'] = df['kecamatan'].str.title().str.strip()
corrections = {
    'Grogol Petamburan': 'Grogolpetamburan',
    'Kebon Jeruk': 'Kebonjeruk',
    'Pal Merah': 'Palmerah',
    'Pasar Rebo': 'Pasarrebo',
    'Taman Sari': 'Tamansari',
    'Tanah Abang': 'Tanahabang',
}
df['kecamatan_final'] = df['kecamatan_clean'].replace(corrections)
df['tahun'] = 2024

# === Train Model ===
features = pd.get_dummies(df[['jumlah_estimasi_penderita',
                              'jumlah_yang_mendapatkan_pelayanan_kesehatan',
                              'wilayah', 'kecamatan', 'jenis_kelamin']], drop_first=True)
target = df['persentase']
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# === Predict 2024–2030 ===
df_all = df[['kecamatan_final', 'wilayah', 'persentase', 'tahun']].copy()
df_all['prioritas'] = df_all['persentase'].apply(lambda x: "Prioritas" if x > 85 else "Tidak")

for year in range(2025, 2031):
    df_future = df.copy()
    df_future['tahun'] = year
    future_features = pd.get_dummies(df_future[['jumlah_estimasi_penderita',
                                                 'jumlah_yang_mendapatkan_pelayanan_kesehatan',
                                                 'wilayah', 'kecamatan', 'jenis_kelamin']], drop_first=True)
    future_features = future_features.reindex(columns=features.columns, fill_value=0)
    df_future['persentase'] = model.predict(future_features)
    df_future['kecamatan_final'] = df_future['kecamatan_clean'].replace(corrections)
    df_future['prioritas'] = df_future['persentase'].apply(lambda x: "Prioritas" if x > 85 else "Tidak")
    df_all = pd.concat([df_all, df_future[['kecamatan_final', 'wilayah', 'persentase', 'tahun', 'prioritas']]])

df_all = df_all.drop_duplicates(subset=['kecamatan_final', 'tahun'])

# === Tambahkan Latitude dan Longitude dari GeoJSON ===
gdf['NAME_3_clean'] = gdf['NAME_3'].str.strip()
gdf = gdf.to_crs(epsg=4326)
gdf['lon'] = gdf.geometry.centroid.x
gdf['lat'] = gdf.geometry.centroid.y
geo_info = gdf[['NAME_3_clean', 'lon', 'lat']]
df_all = df_all.merge(geo_info, left_on='kecamatan_final', right_on='NAME_3_clean', how='left')
df_all.drop(columns=['NAME_3_clean'], inplace=True)

# === Rekomendasi Nakes (Logika Sederhana) ===
df_all['rekomendasi_nakes'] = df_all['persentase'].apply(lambda x: math.ceil(x / 10))

# === Hitung jarak distribusi dari pusat kesehatan ===
pusat_kesehatan = (-6.1767, 106.8306)  # contoh: Monas Jakarta
df_all['jarak'] = df_all.apply(
    lambda row: geodesic(pusat_kesehatan, (row['lat'], row['lon'])).km if pd.notnull(row['lat']) else None,
    axis=1
)

# === Sidebar Filter ===
st.sidebar.title("Filter")
selected_year = st.sidebar.selectbox("Pilih Tahun", sorted(df_all['tahun'].unique()))
selected_wilayah = st.sidebar.selectbox("Pilih Wilayah", ["Semua"] + sorted(df_all['wilayah'].unique()))
selected_kecamatan = st.sidebar.selectbox("Pilih Kecamatan", ["Semua"] + sorted(df_all['kecamatan_final'].unique()))
selected_prioritas = st.sidebar.selectbox("Pilih Status Prioritas", ["Semua", "Prioritas", "Tidak"])

# === Filter Data ===
filtered_df = df_all[df_all['tahun'] == selected_year]
if selected_wilayah != "Semua":
    filtered_df = filtered_df[filtered_df['wilayah'] == selected_wilayah]
if selected_kecamatan != "Semua":
    filtered_df = filtered_df[filtered_df['kecamatan_final'] == selected_kecamatan]
if selected_prioritas != "Semua":
    filtered_df = filtered_df[filtered_df['prioritas'] == selected_prioritas]

# === Merge with GeoJSON ===
merged_gdf = gdf.merge(filtered_df, left_on='NAME_3_clean', right_on='kecamatan_final', how='left')

# === Interactive Map ===
st.subheader(f"Peta Persentase Pelayanan Hipertensi Tahun {selected_year}")
m = folium.Map(location=[-6.2, 106.8], zoom_start=11)

folium.Choropleth(
    geo_data=merged_gdf,
    data=filtered_df,
    columns=['kecamatan_final', 'persentase'],
    key_on='feature.properties.NAME_3',
    fill_color='YlOrRd',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name=f"Persentase Hipertensi Tahun {selected_year}",
    nan_fill_color='white'
).add_to(m)

# === Marker Cluster Gabungan (Persentase + Rekomendasi + Jarak) ===
marker_cluster = MarkerCluster().add_to(m)
for _, row in filtered_df.iterrows():
    if pd.notnull(row['persentase']) and pd.notnull(row['lat']):
        color = 'red' if row['prioritas'] == 'Prioritas' else 'green'
        popup_text = f"""
        <b>{row['kecamatan_final']}</b><br>
        Persentase: {row['persentase']:.2f}%<br>
        Status: {row['prioritas']}<br>
        Rekomendasi Nakes: {int(row['rekomendasi_nakes'])}<br>
        Jarak dari pusat: {row['jarak']:.2f} km
        """
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=7,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=popup_text
        ).add_to(marker_cluster)

# === Jalur Distribusi Puskesmas Keliling ===
st.subheader("Jalur Distribusi Puskesmas Keliling")

tujuan = filtered_df[['kecamatan_final', 'lat', 'lon']].dropna()
if not tujuan.empty:
    tujuan_sorted = tujuan.sort_values(by='kecamatan_final')  # urutan bisa diubah sesuai kebutuhan
    points = [pusat_kesehatan] + list(zip(tujuan_sorted['lat'], tujuan_sorted['lon']))

    folium.PolyLine(
        locations=points,
        color="blue",
        weight=3,
        opacity=0.8,
        tooltip="Jalur Puskesmas Keliling"
    ).add_to(m)

    folium.Marker(
        location=pusat_kesehatan,
        popup="Pusat Kesehatan (Dinkes)",
        icon=folium.Icon(color="red", icon="hospital", prefix="fa")
    ).add_to(m)
else:
    st.info("Tidak ada titik distribusi untuk filter yang dipilih.")

# === Tampilkan Map ===
st_folium(m, width=725)

# === Bar Chart Comparison ===
st.subheader("Perbandingan Persentase dari Tahun ke Tahun")
compare_df = df_all[df_all['kecamatan_final'] == selected_kecamatan] if selected_kecamatan != "Semua" else df_all

fig = px.bar(
    compare_df,
    x='tahun',
    y='persentase',
    color='prioritas',
    barmode='group',
    title=f"Perbandingan Hipertensi {'Kecamatan ' + selected_kecamatan if selected_kecamatan != 'Semua' else 'Semua Kecamatan'}",
    hover_data=['wilayah', 'kecamatan_final', 'rekomendasi_nakes']
)
st.plotly_chart(fig, use_container_width=True)

# === Export to CSV ===
export = df_all.rename(columns={'kecamatan_final': 'kecamatan'})
csv = export[['kecamatan', 'wilayah', 'tahun', 'persentase', 'prioritas', 'rekomendasi_nakes', 'jarak', 'lat', 'lon']].to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Prediksi & Rekomendasi Nakes (2024–2030)", data=csv,
                   file_name='prediksi_hipertensi_prioritas_nakes_koordinat.csv', mime='text/csv')
