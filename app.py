import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Sayfa ayarı
st.set_page_config(page_title="Su Analiz AI", layout="wide")

# Başlık
st.title("🌊 AKILLI SU ANALİZ SİSTEMİ")
st.markdown("---")

# AI Davranış Analizi Fonksiyonları
def calculate_daily_consumption(df, tesisat_no):
    """Günlük ortalama tüketim hesapla"""
    tesisat_data = df[df['TESISAT_NO'] == tesisat_no]
    if len(tesisat_data) == 0:
        return 0
    
    # Okuma tarihleri arasındaki gün sayısı
    dates = pd.to_datetime(tesisat_data['OKUMA_TARIHI'])
    if len(dates) < 2:
        return tesisat_data['AKTIF_m3'].iloc[0] / 30  # Varsayılan 30 gün
    
    days_diff = (dates.max() - dates.min()).days
    if days_diff == 0:
        return tesisat_data['AKTIF_m3'].iloc[0] / 30
    
    total_consumption = tesisat_data['AKTIF_m3'].sum()
    return total_consumption / days_diff

def analyze_consumption_behavior(df, tesisat_no):
    """Tüketim davranışını analiz et"""
    tesisat_data = df[df['TESISAT_NO'] == tesisat_no].sort_values('OKUMA_TARIHI')
    
    if len(tesisat_data) < 2:
        return "Yetersiz veri", "Yetersiz kayıt"
    
    # Tüketim değişkenliği
    consumption_std = tesisat_data['AKTIF_m3'].std()
    consumption_mean = tesisat_data['AKTIF_m3'].mean()
    
    # Anomali tespiti
    anomalies = []
    current_consumption = tesisat_data['AKTIF_m3'].iloc[-1]
    avg_consumption = tesisat_data['AKTIF_m3'].mean()
    
    # Davranış yorumları
    if consumption_std > avg_consumption * 0.5:
        behavior_comment = "Tüketim paterninde olağandışı dalgalanma"
    elif current_consumption == 0 and tesisat_data['TOPLAM_TUTAR'].iloc[-1] > 0:
        behavior_comment = "Su kullanım davranışında farklılaşma gözleniyor"
    elif current_consumption < 5 and tesisat_data['TOPLAM_TUTAR'].iloc[-1] > 100:
        behavior_comment = "Tüketim alışkanlıklarında dikkat çekici değişim"
    elif consumption_std > avg_consumption * 0.3:
        behavior_comment = "Değişken tüketim alışkanlıkları"
    else:
        behavior_comment = "Su kullanım alışkanlıklarında farklılaşma"
    
    # Şüpheli dönemler
    suspicious_periods = "Yok"
    if len(tesisat_data) >= 3:
        high_consumption_periods = tesisat_data[
            tesisat_data['AKTIF_m3'] > avg_consumption * 1.5
        ]
        if len(high_consumption_periods) > 0:
            dates = high_consumption_periods['OKUMA_TARIHI'].dt.strftime('%m/%Y').unique()
            suspicious_periods = ", ".join(dates[:3])  # En fazla 3 dönem göster
    
    return behavior_comment, suspicious_periods

def determine_risk_level(aktif_m3, toplam_tutar, daily_avg):
    """Risk seviyesini belirle"""
    if aktif_m3 == 0 and toplam_tutar > 0:
        return "YÜKSEK"
    elif aktif_m3 <= 5 and toplam_tutar > 100:
        return "YÜKSEK"
    elif daily_avg > 50:  # Çok yüksek günlük tüketim
        return "ORTA"
    elif aktif_m3 <= 15:
        return "ORTA"
    else:
        return "DÜŞÜK"

# Dosya yükleme
uploaded_file = st.file_uploader("📤 yavuz.xlsx dosyasını yükle", type="xlsx")

if uploaded_file:
    # Veriyi oku
    df = pd.read_excel(uploaded_file)
    
    # Tarih düzenleme
    df['OKUMA_TARIHI'] = pd.to_datetime(df['OKUMA_TARIHI'], format='%Y%m%d', errors='coerce')
    df['ILK_OKUMA_TARIHI'] = pd.to_datetime(df['ILK_OKUMA_TARIHI'], format='%Y%m%d', errors='coerce')
    
    # En güncel kayıtları bul
    latest_readings = df.sort_values('OKUMA_TARIHI').groupby('TESISAT_NO').tail(1)
    
    st.success(f"✅ {len(latest_readings)} benzersiz tesisat yüklendi!")
    
    # TAB 1: DETAYLI DAVRANIŞ ANALİZİ
    st.header("📊 TAB 1: DETAYLI DAVRANIŞ ANALİZİ")
    
    # AI analizlerini uygula
    analysis_results = []
    
    for tesisat_no in latest_readings['TESISAT_NO'].unique():
        tesisat_data = latest_readings[latest_readings['TESISAT_NO'] == tesisat_no].iloc[0]
        
        # Metrikleri hesapla
        gunluk_ort_tuketim = calculate_daily_consumption(df, tesisat_no)
        davranis_yorumu, supheli_donemler = analyze_consumption_behavior(df, tesisat_no)
        risk_seviyesi = determine_risk_level(
            tesisat_data['AKTIF_m3'], 
            tesisat_data['TOPLAM_TUTAR'],
            gunluk_ort_tuketim
        )
        
        analysis_results.append({
            'TESISAT_NO': tesisat_no,
            'AKTIF_m3': tesisat_data['AKTIF_m3'],
            'TOPLAM_TUTAR': tesisat_data['TOPLAM_TUTAR'],
            'GUNLUK_ORT_TUKETIM_m3': gunluk_ort_tuketim,
            'DAVRANIS_YORUMU': davranis_yorumu,
            'SUPHELI_DONEMLER': supheli_donemler,
            'RISK_SEVIYESI': risk_seviyesi
        })
    
    # DataFrame oluştur
    detailed_analysis = pd.DataFrame(analysis_results)
    
    # Risk seviyesine göre filtreleme
    risk_filter = st.selectbox(
        "Risk Seviyesi Filtresi:", 
        ["TÜMÜ", "YÜKSEK", "ORTA", "DÜŞÜK"]
    )
    
    if risk_filter != "TÜMÜ":
        filtered_analysis = detailed_analysis[detailed_analysis['RISK_SEVIYESI'] == risk_filter]
    else:
        filtered_analysis = detailed_analysis
    
    # Sıralama
    sort_option = st.selectbox(
        "Sıralama Ölçütü:",
        ["AKTIF_m3 (Azalan)", "TOPLAM_TUTAR (Azalan)", "RISK_SEVIYESI", "GUNLUK_ORT_TUKETIM_m3 (Azalan)"]
    )
    
    if sort_option == "AKTIF_m3 (Azalan)":
        filtered_analysis = filtered_analysis.sort_values('AKTIF_m3', ascending=False)
    elif sort_option == "TOPLAM_TUTAR (Azalan)":
        filtered_analysis = filtered_analysis.sort_values('TOPLAM_TUTAR', ascending=False)
    elif sort_option == "GUNLUK_ORT_TUKETIM_m3 (Azalan)":
        filtered_analysis = filtered_analysis.sort_values('GUNLUK_ORT_TUKETIM_m3', ascending=False)
    else:
        filtered_analysis = filtered_analysis.sort_values('RISK_SEVIYESI')
    
    # Renk kodlu tablo gösterimi
    st.subheader(f"🔍 {risk_filter} Risk Seviyesi - {len(filtered_analysis)} Tesisat")
    
    # DataFrame'i formatla
    display_df = filtered_analysis.copy()
    display_df['GUNLUK_ORT_TUKETIM_m3'] = display_df['GUNLUK_ORT_TUKETIM_m3'].round(6)
    
    # Tabloyu göster
    st.dataframe(
        display_df,
        use_container_width=True,
        column_config={
            "TESISAT_NO": "Tesisat No",
            "AKTIF_m3": "Aktif m³",
            "TOPLAM_TUTAR": "Toplam Tutar",
            "GUNLUK_ORT_TUKETIM_m3": "Günlük Ort. m³",
            "DAVRANIS_YORUMU": "Davranış Yorumu", 
            "SUPHELI_DONEMLER": "Şüpheli Dönemler",
            "RISK_SEVIYESI": "Risk Seviyesi"
        }
    )
    
    # İndirme butonları
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv_all = detailed_analysis.to_csv(index=False, sep='\t')
        st.download_button(
            label="📥 Tüm Analiz Verisi",
            data=csv_all,
            file_name="tum_davranis_analizi.csv",
            mime="text/csv"
        )
    
    with col2:
        csv_high = detailed_analysis[detailed_analysis['RISK_SEVIYESI'] == 'YÜKSEK'].to_csv(index=False, sep='\t')
        st.download_button(
            label="📥 Yüksek Risk Raporu",
            data=csv_high,
            file_name="yuksek_risk_analizi.csv",
            mime="text/csv"
        )
    
    with col3:
        csv_filtered = filtered_analysis.to_csv(index=False, sep='\t')
        st.download_button(
            label=f"📥 {risk_filter} Risk Raporu",
            data=csv_filtered,
            file_name=f"{risk_filter.lower()}_risk_analizi.csv",
            mime="text/csv"
        )
    
    # TAB 2: ÖZET İSTATİSTİKLER
    st.header("📈 TAB 2: ÖZET İSTATİSTİKLER")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Toplam Tesisat", len(detailed_analysis))
        st.metric("Yüksek Risk", len(detailed_analysis[detailed_analysis['RISK_SEVIYESI'] == 'YÜKSEK']))
    
    with col2:
        st.metric("Orta Risk", len(detailed_analysis[detailed_analysis['RISK_SEVIYESI'] == 'ORTA']))
        st.metric("Düşük Risk", len(detailed_analysis[detailed_analysis['RISK_SEVIYESI'] == 'DÜŞÜK']))
    
    with col3:
        total_consumption = detailed_analysis['AKTIF_m3'].sum()
        st.metric("Toplam Tüketim", f"{total_consumption:,.0f} m³")
        avg_daily = detailed_analysis['GUNLUK_ORT_TUKETIM_m3'].mean()
        st.metric("Ort. Günlük Tüketim", f"{avg_daily:.2f} m³")
    
    with col4:
        total_revenue = detailed_analysis['TOPLAM_TUTAR'].sum()
        st.metric("Toplam Gelir", f"{total_revenue:,.0f} TL")
        risk_ratio = (len(detailed_analysis[detailed_analysis['RISK_SEVIYESI'].isin(['YÜKSEK', 'ORTA'])]) / len(detailed_analysis)) * 100
        st.metric("Risk Oranı", f"%{risk_ratio:.1f}")
    
    # TAB 3: ZONE BAZLI ANALİZ
    st.header("🌳 TAB 3: ZONE BAZLI ANALİZ")
    
    # Zone bazlı özet
    zone_summary = latest_readings.groupby('KARNE_NO').agg({
        'TESISAT_NO': 'count',
        'AKTIF_m3': 'sum',
        'TOPLAM_TUTAR': 'sum'
    }).reset_index()
    
    zone_summary.columns = ['ZONE', 'TESISAT_SAYISI', 'TOPLAM_TUKETIM', 'TOPLAM_GELIR']
    
    # Zone bazlı risk analizi
    zone_risk = []
    for zone in zone_summary['ZONE']:
        zone_tesisat = detailed_analysis.merge(
            latest_readings[['TESISAT_NO', 'KARNE_NO']], 
            on='TESISAT_NO'
        )
        zone_data = zone_tesisat[zone_tesisat['KARNE_NO'] == zone]
        
        high_risk_count = len(zone_data[zone_data['RISK_SEVIYESI'] == 'YÜKSEK'])
        medium_risk_count = len(zone_data[zone_data['RISK_SEVIYESI'] == 'ORTA'])
        
        zone_risk.append({
            'ZONE': zone,
            'YUKSEK_RISK': high_risk_count,
            'ORTA_RISK': medium_risk_count
        })
    
    zone_risk_df = pd.DataFrame(zone_risk)
    zone_summary = zone_summary.merge(zone_risk_df, on='ZONE')
    
    # Zone tablosunu göster
    st.dataframe(
        zone_summary,
        use_container_width=True,
        column_config={
            "ZONE": "Zone No",
            "TESISAT_SAYISI": "Tesisat",
            "TOPLAM_TUKETIM": "Toplam m³", 
            "TOPLAM_GELIR": "Toplam TL",
            "YUKSEK_RISK": "🔴 Yüksek",
            "ORTA_RISK": "🟡 Orta"
        }
    )
    
    # Son güncelleme
    st.info(f"🤖 AI Analiz Tarihi: {datetime.now().strftime('%d.%m.%Y %H:%M')} | "
           f"📊 {len(detailed_analysis)} tesisat analiz edildi | "
           f"🎯 {len(filtered_analysis)} {risk_filter} risk seviyesi")

else:
    st.info("👆 Lütfen yavuz.xlsx dosyasını yükleyin")