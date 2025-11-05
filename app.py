import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Streamlit sayfa ayarı
st.set_page_config(
    page_title="Su Tüketim Analiz Dashboard",
    page_icon="💧",
    layout="wide"
)

# Başlık
st.title("💧 Su Tüketim Analiz Dashboard")

# Dosya yükleme
st.sidebar.header("📁 Dosya Yükleme")

uploaded_file = st.sidebar.file_uploader(
    "Ana Excel dosyasını seçin",
    type=["xlsx"],
    help="Su tüketim verilerini içeren Excel dosyasını yükleyin"
)

zone_file = st.sidebar.file_uploader(
    "Zone Excel dosyasını seçin", 
    type=["xlsx"],
    help="Zone bilgilerini içeren Excel dosyasını yükleyin"
)

# Demo butonu
if st.sidebar.button("🎮 Demo Modunda Çalıştır"):
    # Demo verisi oluştur
    st.info("Demo modu aktif! Örnek verilerle çalışılıyor...")
    np.random.seed(42)
    
    # Örnek veri oluştur
    demo_data = []
    for i in range(500):
        tesisat_no = f"TS{1000 + i}"
        aktif_m3 = np.random.gamma(2, 10)
        toplam_tutar = aktif_m3 * 15
        
        demo_data.append({
            'TESISAT_NO': tesisat_no,
            'AKTIF_m3': max(aktif_m3, 0.1),
            'TOPLAM_TUTAR': max(toplam_tutar, 0),
            'ILK_OKUMA_TARIHI': pd.Timestamp('2023-01-01'),
            'OKUMA_TARIHI': pd.Timestamp('2024-10-31'),
            'KARNE_NO': np.random.choice(['8050', '8055', '8060', '8065', '8070'])
        })
    
    df = pd.DataFrame(demo_data)
    
    # Basit analiz
    df['OKUMA_PERIYODU_GUN'] = 300
    df['GUNLUK_ORT_TUKETIM_m3'] = df['AKTIF_m3'] / df['OKUMA_PERIYODU_GUN']
    
    # Risk seviyesi
    def risk_hesapla(tuketim):
        if tuketim > 50:
            return 'Yüksek'
        elif tuketim > 20:
            return 'Orta'
        else:
            return 'Düşük'
    
    df['RISK_SEVIYESI'] = df['AKTIF_m3'].apply(risk_hesapla)
    
    # Zone analizi
    zone_analizi = df.groupby('KARNE_NO').agg({
        'TESISAT_NO': 'count',
        'AKTIF_m3': 'sum',
        'TOPLAM_TUTAR': 'sum'
    }).reset_index()
    
    zone_analizi.columns = ['KARNE_NO', 'TESISAT_SAYISI', 'TOPLAM_TUKETIM', 'TOPLAM_GELIR']
    
    # Zone bilgileri
    kullanici_zone_verileri = {
        '8050': {'ad': 'ÖLÇÜM NOKTASI-5 (ÜST BÖLGE)', 'verilen_su': 18666, 'tahakkuk_m3': 7654, 'kayip_oran': 58.99},
        '8055': {'ad': 'ÖLÇÜM NOKTASI-3 (ALT BÖLGE)', 'verilen_su': 19623, 'tahakkuk_m3': 7375, 'kayip_oran': 62.42},
        '8060': {'ad': 'ÖLÇÜM NOKTASI-1 (KIRMIZI)', 'verilen_su': 20078, 'tahakkuk_m3': 7010, 'kayip_oran': 65.09},
        '8065': {'ad': 'ÖLÇÜM NOKTASI-2 (MAVİ)', 'verilen_su': 3968, 'tahakkuk_m3': 1813, 'kayip_oran': 54.31},
        '8070': {'ad': 'HASTANE BÖLGESİ', 'verilen_su': 17775, 'tahakkuk_m3': 2134, 'kayip_oran': 87.99}
    }
    
    st.success("✅ Demo verisi başarıyla oluşturuldu!")
    
    # Verileri global değişkenlere ata
    son_okumalar = df
    zone_analizi = zone_analizi

elif uploaded_file is not None:
    try:
        # Gerçek dosya yüklendi
        df = pd.read_excel(uploaded_file)
        st.success(f"✅ Ana veri başarıyla yüklendi: {len(df)} kayıt")
        
        # Tarih formatını düzelt
        if 'ILK_OKUMA_TARIHI' in df.columns:
            df['ILK_OKUMA_TARIHI'] = pd.to_datetime(df['ILK_OKUMA_TARIHI'], errors='coerce')
        if 'OKUMA_TARIHI' in df.columns:
            df['OKUMA_TARIHI'] = pd.to_datetime(df['OKUMA_TARIHI'], errors='coerce')
        
        # Zone dosyasını oku
        kullanici_zone_verileri = {}
        if zone_file is not None:
            try:
                zone_df = pd.read_excel(zone_file)
                st.success(f"✅ Zone verisi başarıyla yüklendi: {len(zone_df)} kayıt")
                
                # Basit zone verisi işleme
                for idx, row in zone_df.iterrows():
                    if 'KARNE NO VE ADI' in zone_df.columns:
                        karne_adi = str(row['KARNE NO VE ADI'])
                        # Basit karne no çıkarma
                        import re
                        karne_no_match = re.search(r'(\d{4})', karne_adi)
                        if karne_no_match:
                            karne_no = karne_no_match.group(1)
                            kullanici_zone_verileri[karne_no] = {
                                'ad': karne_adi,
                                'verilen_su': row.get('VERİLEN SU MİKTARI M3', 0),
                                'tahakkuk_m3': row.get('TAHAKKUK M3', 0),
                                'kayip_oran': row.get('BRÜT KAYIP KAÇAK ORANI\n%', 0)
                            }
            except Exception as e:
                st.warning(f"Zone dosyası işlenirken hata: {e}")
        
        # Basit analiz
        son_okumalar = df.sort_values('OKUMA_TARIHI').groupby('TESISAT_NO').last().reset_index()
        
        if 'ILK_OKUMA_TARIHI' in son_okumalar.columns and 'OKUMA_TARIHI' in son_okumalar.columns:
            son_okumalar['OKUMA_PERIYODU_GUN'] = (son_okumalar['OKUMA_TARIHI'] - son_okumalar['ILK_OKUMA_TARIHI']).dt.days
            son_okumalar['OKUMA_PERIYODU_GUN'] = son_okumalar['OKUMA_PERIYODU_GUN'].clip(lower=1, upper=365)
        else:
            son_okumalar['OKUMA_PERIYODU_GUN'] = 30
        
        son_okumalar['GUNLUK_ORT_TUKETIM_m3'] = son_okumalar['AKTIF_m3'] / son_okumalar['OKUMA_PERIYODU_GUN']
        
        # Risk seviyesi
        def risk_hesapla(tuketim):
            if tuketim > 50:
                return 'Yüksek'
            elif tuketim > 20:
                return 'Orta'
            else:
                return 'Düşük'
        
        son_okumalar['RISK_SEVIYESI'] = son_okumalar['AKTIF_m3'].apply(risk_hesapla)
        
        # Zone analizi
        if 'KARNE_NO' in df.columns:
            zone_analizi = df.groupby('KARNE_NO').agg({
                'TESISAT_NO': 'count',
                'AKTIF_m3': 'sum',
                'TOPLAM_TUTAR': 'sum'
            }).reset_index()
            zone_analizi.columns = ['KARNE_NO', 'TESISAT_SAYISI', 'TOPLAM_TUKETIM', 'TOPLAM_GELIR']
        else:
            zone_analizi = None
            
    except Exception as e:
        st.error(f"❌ Dosya işleme hatası: {e}")
        st.stop()
else:
    st.warning("⚠️ Lütfen Excel dosyalarını yükleyin veya Demo modunu kullanın")
    st.stop()

# Metrikler
if 'son_okumalar' in locals():
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Toplam Tesisat", f"{len(son_okumalar):,}")
    
    with col2:
        toplam_tuketim = son_okumalar['AKTIF_m3'].sum()
        st.metric("💧 Toplam Tüketim", f"{toplam_tuketim:,.0f} m³")
    
    with col3:
        toplam_gelir = son_okumalar['TOPLAM_TUTAR'].sum() if 'TOPLAM_TUTAR' in son_okumalar.columns else 0
        st.metric("💰 Toplam Gelir", f"{toplam_gelir:,.0f} TL")
    
    with col4:
        yuksek_riskli = len(son_okumalar[son_okumalar['RISK_SEVIYESI'] == 'Yüksek'])
        st.metric("🚨 Yüksek Riskli", f"{yuksek_riskli}")

# Tablar
tab1, tab2, tab3 = st.tabs(["📈 Genel Görünüm", "🗺️ Zone Analizi", "🔍 Detaylı Analiz"])

with tab1:
    if 'son_okumalar' in locals():
        col1, col2 = st.columns(2)
        
        with col1:
            # Tüketim dağılımı
            fig1 = px.histogram(son_okumalar, x='AKTIF_m3', 
                              title='Tüketim Dağılımı',
                              labels={'AKTIF_m3': 'Tüketim (m³)'})
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Risk dağılımı
            risk_dagilim = son_okumalar['RISK_SEVIYESI'].value_counts()
            fig2 = px.pie(values=risk_dagilim.values, names=risk_dagilim.index,
                         title='Risk Seviyeleri Dağılımı',
                         color_discrete_map={'Yüksek': 'red', 'Orta': 'orange', 'Düşük': 'green'})
            st.plotly_chart(fig2, use_container_width=True)

with tab2:
    if 'zone_analizi' in locals() and zone_analizi is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            # Zone tüketim dağılımı
            fig3 = px.pie(zone_analizi, values='TOPLAM_TUKETIM', names='KARNE_NO',
                         title='Zone Bazlı Tüketim Dağılımı')
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            # Zone tesisat sayısı
            fig4 = px.bar(zone_analizi, x='KARNE_NO', y='TESISAT_SAYISI',
                         title='Zone Bazlı Tesisat Sayısı')
            st.plotly_chart(fig4, use_container_width=True)
        
        # Zone tablosu
        st.subheader("Zone Karşılaştırma Tablosu")
        st.dataframe(zone_analizi, use_container_width=True)

with tab3:
    if 'son_okumalar' in locals():
        st.subheader("Tesisat Detayları")
        
        # Filtreleme
        col1, col2 = st.columns(2)
        
        with col1:
            risk_filtre = st.multiselect(
                "Risk Seviyesi",
                options=['Düşük', 'Orta', 'Yüksek'],
                default=['Yüksek', 'Orta']
            )
        
        with col2:
            siralama = st.selectbox(
                "Sıralama",
                options=['Yüksek Risk', 'Yüksek Tüketim', 'Düşük Tüketim']
            )
        
        # Filtrele
        filtreli = son_okumalar[son_okumalar['RISK_SEVIYESI'].isin(risk_filtre)]
        
        # Sırala
        if siralama == 'Yüksek Risk':
            filtreli = filtreli.sort_values('RISK_SEVIYESI', ascending=False)
        elif siralama == 'Yüksek Tüketim':
            filtreli = filtreli.sort_values('AKTIF_m3', ascending=False)
        else:
            filtreli = filtreli.sort_values('AKTIF_m3', ascending=True)
        
        # Göster
        st.dataframe(filtreli[['TESISAT_NO', 'AKTIF_m3', 'TOPLAM_TUTAR', 'GUNLUK_ORT_TUKETIM_m3', 'RISK_SEVIYESI']].head(20), 
                    use_container_width=True)

# Footer
st.markdown("---")
st.markdown("💧 Su Tüketim Analiz Sistemi | Streamlit Dashboard")
