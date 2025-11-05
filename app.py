import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
import warnings
import re
warnings.filterwarnings('ignore')

# ======================================================================
# 🚀 STREAMLIT UYGULAMASI
# ======================================================================

st.set_page_config(
    page_title="Su Tüketim Davranış Analiz Dashboard",
    page_icon="💧",
    layout="wide"
)

st.title("💧 Su Tüketim Davranış Analiz Dashboard")
st.markdown("**Profesyonel Analiz Sistemi | Gelişmiş Risk Tespiti | Zone Bazlı Raporlama**")

# ======================================================================
# 📊 VERİ İŞLEME FONKSİYONLARI
# ======================================================================

@st.cache_data
def load_and_analyze_data(uploaded_file, zone_file):
    """İki dosyadan veriyi okur ve profesyonel analiz eder"""
    try:
        # Ana veri dosyasını oku
        df = pd.read_excel(uploaded_file)
        st.success(f"✅ Ana veri başarıyla yüklendi: {len(df)} kayıt")
    except Exception as e:
        st.error(f"❌ Ana dosya okuma hatası: {e}")
        return None, None, None, None

    # Tarih formatını düzelt
    df['ILK_OKUMA_TARIHI'] = pd.to_datetime(df['ILK_OKUMA_TARIHI'], format='%Y%m%d', errors='coerce')
    df['OKUMA_TARIHI'] = pd.to_datetime(df['OKUMA_TARIHI'], format='%Y%m%d', errors='coerce')

    # Tesisat numarası olan kayıtları filtrele
    df = df[df['TESISAT_NO'].notnull()]

    # Zone veri dosyasını oku
    kullanici_zone_verileri = {}
    if zone_file is not None:
        try:
            zone_excel_df = pd.read_excel(zone_file)
            st.success(f"✅ Zone veri dosyası başarıyla yüklendi: {len(zone_excel_df)} kayıt")
            
            # Zone verilerini işle
            for idx, row in zone_excel_df.iterrows():
                # Karne no ve adını ayır
                if 'KARNE NO VE ADI' in row:
                    karne_adi = str(row['KARNE NO VE ADI']).strip()
                    
                    # Karne numarasını çıkar (ilk 4 rakam)
                    karne_no_match = re.search(r'(\d{4})', karne_adi)
                    if karne_no_match:
                        karne_no = karne_no_match.group(1)
                        
                        # Zone bilgilerini topla
                        zone_bilgisi = {
                            'ad': karne_adi,
                            'verilen_su': row.get('VERİLEN SU MİKTARI M3', 0),
                            'tahakkuk_m3': row.get('TAHAKKUK M3', 0),
                            'kayip_oran': row.get('BRÜT KAYIP KAÇAK ORANI\n%', 0)
                        }
                        
                        kullanici_zone_verileri[karne_no] = zone_bilgisi
        except Exception as e:
            st.error(f"❌ Zone veri dosyası yüklenirken hata: {e}")

    # ======================================================================
    # 🔍 VERİ KALİTESİ ANALİZİ
    # ======================================================================
    
    st.subheader("🔍 Veri Kalitesi Analizi")
    
    # Eksik veri kontrolü
    eksik_veri_analizi = pd.DataFrame({
        'SUTUN': df.columns,
        'EKSIK_SAYISI': df.isnull().sum(),
        'EKSIK_ORANI (%)': (df.isnull().sum() / len(df)) * 100
    })
    
    eksik_sutunlar = eksik_veri_analizi[eksik_veri_analizi['EKSIK_SAYISI'] > 0]
    
    if len(eksik_sutunlar) > 0:
        st.warning(f"⚠️ {len(eksik_sutunlar)} sütunda eksik veri bulundu")
        st.dataframe(eksik_sutunlar[['SUTUN', 'EKSIK_SAYISI', 'EKSIK_ORANI (%)']])
    else:
        st.success("✅ Hiç eksik veri bulunamadı")

    # ======================================================================
    # 🔍 TESİSAT BAZLI DAVRANIŞ ANALİZİ
    # ======================================================================

    # Her tesisat için en son okuma kaydını al
    son_okumalar = df.sort_values('OKUMA_TARIHI').groupby('TESISAT_NO').last().reset_index()

    # Okuma periyodunu hesapla
    son_okumalar['OKUMA_PERIYODU_GUN'] = (son_okumalar['OKUMA_TARIHI'] - son_okumalar['ILK_OKUMA_TARIHI']).dt.days
    son_okumalar['OKUMA_PERIYODU_GUN'] = son_okumalar['OKUMA_PERIYODU_GUN'].clip(lower=1, upper=365)

    # Günlük ortalama tüketim
    son_okumalar['GUNLUK_ORT_TUKETIM_m3'] = son_okumalar['AKTIF_m3'] / son_okumalar['OKUMA_PERIYODU_GUN']
    son_okumalar['GUNLUK_ORT_TUKETIM_m3'] = son_okumalar['GUNLUK_ORT_TUKETIM_m3'].clip(lower=0.001, upper=100)

    # ======================================================================
    # 🎭 DAVRANIŞ ANALİZİ FONKSİYONLARI
    # ======================================================================

    def tesisat_davranis_analizi(tesisat_no, son_okuma_row):
        """Tesisatın tüketim davranışını analiz et"""
        tesisat_verisi = df[df['TESISAT_NO'] == tesisat_no].sort_values('OKUMA_TARIHI')
        
        if len(tesisat_verisi) < 3:
            return "Yetersiz veri", "Yetersiz kayıt", "Orta"
        
        tuketimler = tesisat_verisi['AKTIF_m3'].values
        tarihler_series = tesisat_verisi['OKUMA_TARIHI']
        
        # Sıfır tüketim analizi
        sifir_sayisi = sum(tuketimler == 0)
        
        # Varyasyon analizi
        std_dev = np.std(tuketimler) if len(tuketimler) > 1 else 0
        mean_tuketim = np.mean(tuketimler) if len(tuketimler) > 0 else 0
        varyasyon_katsayisi = std_dev / mean_tuketim if mean_tuketim > 0 else 0
        
        # Trend analizi (son 3 dönem)
        if len(tuketimler) >= 3:
            son_uc = tuketimler[-3:]
            trend = "artış" if son_uc[2] > son_uc[0] * 1.2 else "azalış" if son_uc[2] < son_uc[0] * 0.8 else "stabil"
        else:
            trend = "belirsiz"
        
        # Şüpheli durum tespiti ve risk seviyesi
        suphe_aciklamasi = ""
        suphe_donemleri = []
        risk_seviyesi = "Düşük"
        
        # 1. Düzensiz sıfır tüketim paterni
        if sifir_sayisi >= 3:
            sifir_indisler = np.where(tuketimler == 0)[0]
            if len(sifir_indisler) >= 3:
                ardisik_olmayan = sum(np.diff(sifir_indisler) > 1) >= 2
                if ardisik_olmayan:
                    suphe_aciklamasi += "Düzensiz sıfır tüketim paterni. "
                    risk_seviyesi = "Yüksek"
                    for idx in sifir_indisler:
                        tarih_obj = pd.Timestamp(tarihler_series.iloc[idx])
                        suphe_donemleri.append(tarih_obj.strftime('%m/%Y'))
        
        # 2. Ani tüketim değişiklikleri
        if varyasyon_katsayisi > 1.5 and mean_tuketim > 5:
            suphe_aciklamasi += "Tüketimde yüksek dalgalanma. "
            risk_seviyesi = "Orta" if risk_seviyesi == "Düşük" else risk_seviyesi
        
        # 3. Trend analizi
        if trend == "artış" and mean_tuketim > 20:
            suphe_aciklamasi += "Yükselen tüketim trendi. "
            risk_seviyesi = "Orta" if risk_seviyesi == "Düşük" else risk_seviyesi
        
        # 4. Son dönem sıfır tüketim
        if tuketimler[-1] == 0 and len(tuketimler) > 1:
            suphe_aciklamasi += "Son dönem sıfır tüketim. "
            risk_seviyesi = "Yüksek" if sifir_sayisi >= 2 else "Orta"
        
        # Şüpheli dönemler varsa risk en az Orta olmalı
        if suphe_donemleri and risk_seviyesi == "Düşük":
            risk_seviyesi = "Orta"
        
        # Yorum kütüphanesi
        yorumlar_normal = ["Normal tüketim paterni", "Stabil tüketim alışkanlığı"]
        yorumlar_supheli = [
            "Tüketim alışkanlıklarında değişiklik gözlemleniyor",
            "Düzensiz tüketim paterni dikkat çekici",
            "Tüketim davranışında tutarsızlık mevcut",
            "Değişken tüketim alışkanlıkları",
            "Tüketim paterninde olağandışı dalgalanma"
        ]
        
        if not suphe_aciklamasi:
            davranis_yorumu = np.random.choice(yorumlar_normal)
        else:
            davranis_yorumu = np.random.choice(yorumlar_supheli)
        
        return davranis_yorumu, ", ".join(suphe_donemleri) if suphe_donemleri else "Yok", risk_seviyesi

    # ======================================================================
    # 📊 DAVRANIŞ BAZLI RAPOR
    # ======================================================================

    st.info("🔍 Tesisat davranış analizleri yapılıyor...")
    progress_bar = st.progress(0)
    davranis_sonuclari = []
    
    total_tesisat = len(son_okumalar)
    for i, (idx, row) in enumerate(son_okumalar.iterrows()):
        yorum, supheli_donemler, risk = tesisat_davranis_analizi(row['TESISAT_NO'], row)
        davranis_sonuclari.append({
            'TESISAT_NO': row['TESISAT_NO'],
            'DAVRANIS_YORUMU': yorum,
            'SUPHELI_DONEMLER': supheli_donemler,
            'RISK_SEVIYESI': risk
        })
        
        if i % 100 == 0:
            progress_bar.progress(min((i + 1) / total_tesisat, 1.0))

    progress_bar.progress(1.0)
    davranis_df = pd.DataFrame(davranis_sonuclari)
    son_okumalar = son_okumalar.merge(davranis_df, on='TESISAT_NO', how='left')

    # ======================================================================
    # 🗺️ ZONE ANALİZİ
    # ======================================================================

    zone_analizi = None
    if 'KARNE_NO' in df.columns:
        # Ekim 2024 verilerini filtrele
        ekim_2024_df = df[(df['OKUMA_TARIHI'].dt.month == 10) & (df['OKUMA_TARIHI'].dt.year == 2024)]
        
        if len(ekim_2024_df) == 0:
            ekim_2024_df = df.copy()
        
        # Zone bazlı analiz
        zone_analizi = ekim_2024_df.groupby('KARNE_NO').agg({
            'TESISAT_NO': 'count',
            'AKTIF_m3': 'sum',
            'TOPLAM_TUTAR': 'sum'
        }).reset_index()
        
        zone_analizi.columns = ['KARNE_NO', 'TESISAT_SAYISI', 'TOPLAM_TUKETIM', 'TOPLAM_GELIR']

        # Zone bazlı risk analizi
        ekim_2024_risk = ekim_2024_df.merge(son_okumalar[['TESISAT_NO', 'RISK_SEVIYESI']], on='TESISAT_NO', how='left')
        zone_risk_analizi = ekim_2024_risk.groupby('KARNE_NO')['RISK_SEVIYESI'].apply(
            lambda x: (x == 'Yüksek').sum() if 'Yüksek' in x.values else 0
        ).reset_index(name='YUKSEK_RISKLI_TESISAT')
        
        zone_analizi = zone_analizi.merge(zone_risk_analizi, on='KARNE_NO', how='left')
        zone_analizi['YUKSEK_RISK_ORANI'] = (zone_analizi['YUKSEK_RISKLI_TESISAT'] / zone_analizi['TESISAT_SAYISI']) * 100

        # Kullanıcı zone verilerini birleştir
        if kullanici_zone_verileri:
            zone_analizi['KARNE_NO'] = zone_analizi['KARNE_NO'].astype(str)
            kullanici_df = pd.DataFrame.from_dict(kullanici_zone_verileri, orient='index').reset_index()
            kullanici_df = kullanici_df.rename(columns={'index': 'KARNE_NO'})
            zone_analizi = zone_analizi.merge(kullanici_df, on='KARNE_NO', how='left')

    return df, son_okumalar, zone_analizi, kullanici_zone_verileri

# ======================================================================
# 🎨 STREAMLIT ARAYÜZ
# ======================================================================

# Dosya yükleme bölümü
st.sidebar.header("📁 İki Dosya Yükle")

uploaded_file = st.sidebar.file_uploader(
    "Ana Excel dosyasını seçin (yavuz.xlsx)",
    type=["xlsx"],
    help="Su tüketim verilerini içeren Excel dosyasını yükleyin"
)

zone_file = st.sidebar.file_uploader(
    "Zone Excel dosyasını seçin (yavuzeli merkez ekim.xlsx)",
    type=["xlsx"],
    help="Zone bilgilerini içeren Excel dosyasını yükleyin"
)

# Demo butonu
if st.sidebar.button("🎮 Demo Modunda Çalıştır"):
    st.info("Demo modu aktif! Profesyonel analiz ile çalışılıyor...")
    np.random.seed(42)
    
    # Gerçekçi demo verisi oluştur
    demo_data = []
    for i in range(2000):
        tesisat_no = f"TS{10000 + i}"
        
        # Farklı tüketim patternleri
        pattern_type = np.random.choice(['normal', 'sifir_aralikli', 'yuksek_dalgalanma'], p=[0.7, 0.2, 0.1])
        
        if pattern_type == 'normal':
            aktif_m3 = np.random.gamma(2, 8)
        elif pattern_type == 'sifir_aralikli':
            aktif_m3 = 0 if np.random.random() < 0.4 else np.random.gamma(2, 6)
        else:
            aktif_m3 = np.random.gamma(4, 12)
        
        toplam_tutar = aktif_m3 * 15 + np.random.normal(0, 10)
        
        demo_data.append({
            'TESISAT_NO': tesisat_no,
            'AKTIF_m3': max(aktif_m3, 0),
            'TOPLAM_TUTAR': max(toplam_tutar, 0),
            'ILK_OKUMA_TARIHI': pd.Timestamp('2023-01-01'),
            'OKUMA_TARIHI': pd.Timestamp('2024-10-31'),
            'KARNE_NO': f"80{np.random.randint(50, 71)}"
        })
    
    df = pd.DataFrame(demo_data)
    
    # Davranış analizi
    son_okumalar = df.copy()
    son_okumalar['OKUMA_PERIYODU_GUN'] = 300
    son_okumalar['GUNLUK_ORT_TUKETIM_m3'] = son_okumalar['AKTIF_m3'] / son_okumalar['OKUMA_PERIYODU_GUN']
    
    # Risk dağılımı
    risk_dagilimi = np.random.choice(['Düşük', 'Orta', 'Yüksek'], size=len(son_okumalar), p=[0.6, 0.3, 0.1])
    son_okumalar['RISK_SEVIYESI'] = risk_dagilimi
    son_okumalar['DAVRANIS_YORUMU'] = "Demo verisi analiz edildi"
    son_okumalar['SUPHELI_DONEMLER'] = "Yok"
    
    # Zone analizi
    zone_analizi = df.groupby('KARNE_NO').agg({
        'TESISAT_NO': 'count',
        'AKTIF_m3': 'sum',
        'TOPLAM_TUTAR': 'sum'
    }).reset_index()
    zone_analizi.columns = ['KARNE_NO', 'TESISAT_SAYISI', 'TOPLAM_TUKETIM', 'TOPLAM_GELIR']
    
    # Risk analizi
    zone_risk = son_okumalar.groupby('KARNE_NO')['RISK_SEVIYESI'].apply(
        lambda x: (x == 'Yüksek').sum()
    ).reset_index(name='YUKSEK_RISKLI_TESISAT')
    
    zone_analizi = zone_analizi.merge(zone_risk, on='KARNE_NO', how='left')
    zone_analizi['YUKSEK_RISK_ORANI'] = (zone_analizi['YUKSEK_RISKLI_TESISAT'] / zone_analizi['TESISAT_SAYISI']) * 100
    
    # Zone bilgileri
    kullanici_zone_verileri = {
        '8050': {'ad': 'ÖLÇÜM NOKTASI-5 (ÜST BÖLGE) (MOR)', 'verilen_su': 18666.00, 'tahakkuk_m3': 7654.00, 'kayip_oran': 58.99},
        '8055': {'ad': 'ÖLÇÜM NOKTASI-3 (ALT BÖLGE) (YEŞİL)', 'verilen_su': 19623.00, 'tahakkuk_m3': 7375.00, 'kayip_oran': 62.42},
        '8060': {'ad': 'ÖLÇÜM NOKTASI-1 (KIRMIZI)', 'verilen_su': 20078.00, 'tahakkuk_m3': 7010.00, 'kayip_oran': 65.09},
        '8065': {'ad': 'ÖLÇÜM NOKTASI-2 (MAVİ)', 'verilen_su': 3968.00, 'tahakkuk_m3': 1813.00, 'kayip_oran': 54.31},
        '8070': {'ad': 'HASTANE BÖLGESİ (SARI)', 'verilen_su': 17775.00, 'tahakkuk_m3': 2134.00, 'kayip_oran': 87.99}
    }
    
    st.success("✅ Profesyonel demo verisi başarıyla oluşturuldu!")

elif uploaded_file is not None:
    # Gerçek dosya yüklendi
    df, son_okumalar, zone_analizi, kullanici_zone_verileri = load_and_analyze_data(uploaded_file, zone_file)
else:
    st.warning("⚠️ Lütfen Excel dosyalarını yükleyin veya Demo modunu kullanın")
    st.stop()

# ======================================================================
# 📊 GENEL METRİKLER
# ======================================================================

st.header("📊 Genel Metrikler")

if son_okumalar is not None:
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Toplam Tesisat", f"{len(son_okumalar):,}")
    
    with col2:
        st.metric("Toplam Tüketim", f"{son_okumalar['AKTIF_m3'].sum():,.0f} m³")
    
    with col3:
        st.metric("Toplam Gelir", f"{son_okumalar['TOPLAM_TUTAR'].sum():,.0f} TL")
    
    with col4:
        yuksek_riskli = len(son_okumalar[son_okumalar['RISK_SEVIYESI'] == 'Yüksek'])
        st.metric("Yüksek Riskli", f"{yuksek_riskli}")
    
    with col5:
        orta_riskli = len(son_okumalar[son_okumalar['RISK_SEVIYESI'] == 'Orta'])
        st.metric("Orta Riskli", f"{orta_riskli}")

# ======================================================================
# 📈 TAB MENÜ
# ======================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Risk Analizi", 
    "🗺️ Zone Analizi", 
    "🔍 Detaylı İnceleme",
    "📋 Özet Rapor"
])

with tab1:
    st.header("📊 Risk Dağılımı Analizi")
    
    if son_okumalar is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk dağılımı pasta grafiği
            risk_dagilim = son_okumalar['RISK_SEVIYESI'].value_counts()
            fig1 = px.pie(values=risk_dagilim.values, names=risk_dagilim.index,
                         title='Risk Seviyeleri Dağılımı',
                         color_discrete_map={'Yüksek': 'red', 'Orta': 'orange', 'Düşük': 'green'})
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Tüketim dağılımı (risk renkli)
            fig2 = px.histogram(son_okumalar, x='AKTIF_m3', color='RISK_SEVIYESI',
                              title='Tüketim Dağılımı (Risk Seviyeli)',
                              labels={'AKTIF_m3': 'Tüketim (m³)'},
                              color_discrete_map={'Yüksek': 'red', 'Orta': 'orange', 'Düşük': 'green'})
            st.plotly_chart(fig2, use_container_width=True)
        
        # Yüksek riskli tesisatlar
        st.subheader("🚨 Yüksek Riskli Tesisatlar")
        yuksek_riskli = son_okumalar[son_okumalar['RISK_SEVIYESI'] == 'Yüksek']
        
        if len(yuksek_riskli) > 0:
            st.dataframe(
                yuksek_riskli[['TESISAT_NO', 'AKTIF_m3', 'TOPLAM_TUTAR', 'GUNLUK_ORT_TUKETIM_m3', 'DAVRANIS_YORUMU', 'SUPHELI_DONEMLER']].head(20),
                use_container_width=True
            )
        else:
            st.success("✅ Yüksek riskli tesisat bulunamadı")

with tab2:
    st.header("🗺️ Zone Bazlı Analiz")
    
    if zone_analizi is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            # Zone tüketim dağılımı
            fig3 = px.pie(zone_analizi, values='TOPLAM_TUKETIM', names='KARNE_NO',
                         title='Zone Bazlı Tüketim Dağılımı')
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            # Zone risk oranları
            fig4 = px.bar(zone_analizi, x='KARNE_NO', y='YUKSEK_RISK_ORANI',
                         title='Zone Bazlı Yüksek Risk Oranları',
                         labels={'KARNE_NO': 'Zone', 'YUKSEK_RISK_ORANI': 'Yüksek Risk Oranı (%)'},
                         color='YUKSEK_RISK_ORANI',
                         color_continuous_scale='reds')
            st.plotly_chart(fig4, use_container_width=True)
        
        # Zone karşılaştırma tablosu
        st.subheader("Zone Karşılaştırma Tablosu")
        zone_karsilastirma = zone_analizi[['KARNE_NO', 'TESISAT_SAYISI', 'TOPLAM_TUKETIM', 'TOPLAM_GELIR', 'YUKSEK_RISK_ORANI']].copy()
        
        if 'ad' in zone_analizi.columns:
            zone_karsilastirma['Zone Adı'] = zone_analizi['ad']
        if 'verilen_su' in zone_analizi.columns:
            zone_karsilastirma['Verilen Su (m³)'] = zone_analizi['verilen_su']
            zone_karsilastirma['Tahakkuk (m³)'] = zone_analizi['tahakkuk_m3']
            zone_karsilastirma['Kayıp Oranı (%)'] = zone_analizi['kayip_oran']
        
        st.dataframe(zone_karsilastirma, use_container_width=True)
    else:
        st.info("Zone verisi bulunamadı")

with tab3:
    st.header("🔍 Detaylı İnceleme")
    
    if son_okumalar is not None:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Filtreleme Seçenekleri")
            
            # Risk seviyesi filtresi
            risk_seviyeleri = st.multiselect(
                "Risk Seviyeleri",
                options=['Düşük', 'Orta', 'Yüksek'],
                default=['Yüksek', 'Orta']
            )
            
            # Tüketim aralığı
            min_tuketim = st.number_input("Min Tüketim (m³)", value=0)
            max_tuketim = st.number_input("Max Tüketim (m³)", value=100)
            
            # Sıralama
            siralama = st.selectbox(
                "Sıralama",
                options=['Yüksek Risk', 'Yüksek Tüketim', 'Düşük Tüketim']
            )
        
        with col2:
            st.subheader("Tesisat Tablosu")
            
            # Filtreleme
            filtreli_veri = son_okumalar[
                (son_okumalar['RISK_SEVIYESI'].isin(risk_seviyeleri)) &
                (son_okumalar['AKTIF_m3'] >= min_tuketim) &
                (son_okumalar['AKTIF_m3'] <= max_tuketim)
            ]
            
            # Sıralama
            if siralama == 'Yüksek Risk':
                risk_sirasi = {'Yüksek': 3, 'Orta': 2, 'Düşük': 1}
                filtreli_veri['RISK_SIRASI'] = filtreli_veri['RISK_SEVIYESI'].map(risk_sirasi)
                filtreli_veri = filtreli_veri.sort_values(['RISK_SIRASI', 'AKTIF_m3'], ascending=[False, False])
            elif siralama == 'Yüksek Tüketim':
                filtreli_veri = filtreli_veri.sort_values('AKTIF_m3', ascending=False)
            else:
                filtreli_veri = filtreli_veri.sort_values('AKTIF_m3', ascending=True)
            
            st.dataframe(
                filtreli_veri[['TESISAT_NO', 'AKTIF_m3', 'TOPLAM_TUTAR', 'GUNLUK_ORT_TUKETIM_m3', 'RISK_SEVIYESI', 'DAVRANIS_YORUMU']].head(50),
                use_container_width=True
            )

with tab4:
    st.header("📋 Özet Rapor")
    
    if son_okumalar is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Genel Özet")
            
            toplam_tesisat = len(son_okumalar)
            toplam_tuketim = son_okumalar['AKTIF_m3'].sum()
            toplam_gelir = son_okumalar['TOPLAM_TUTAR'].sum()
            
            st.metric("Toplam Tesisat", f"{toplam_tesisat:,}")
            st.metric("Toplam Tüketim", f"{toplam_tuketim:,.0f} m³")
            st.metric("Toplam Gelir", f"{toplam_gelir:,.0f} TL")
            
            # Risk dağılımı
            risk_dagilim = son_okumalar['RISK_SEVIYESI'].value_counts()
            st.write("**Risk Dağılımı:**")
            for risk, sayi in risk_dagilim.items():
                yuzde = (sayi / toplam_tesisat) * 100
                st.write(f"- {risk}: {sayi} tesisat (%{yuzde:.1f})")
        
        with col2:
            st.subheader("Finansal Analiz")
            
            # Risk seviyelerine göre gelir analizi
            risk_gelir_analizi = son_okumalar.groupby('RISK_SEVIYESI').agg({
                'TESISAT_NO': 'count',
                'TOPLAM_TUTAR': 'sum',
                'AKTIF_m3': 'sum'
            }).reset_index()
            
            if toplam_gelir > 0:
                risk_gelir_analizi['GELIR_PAYI'] = (risk_gelir_analizi['TOPLAM_TUTAR'] / toplam_gelir) * 100
            
            st.dataframe(risk_gelir_analizi, use_container_width=True)
        
        # İndirme butonu
        st.subheader("Rapor İndir")
        
        # Excel raporu oluştur
        @st.cache_data
        def convert_df_to_excel(df):
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Tüm_Tesisatlar')
            processed_data = output.getvalue()
            return processed_data
        
        excel_data = convert_df_to_excel(son_okumalar)
        
        st.download_button(
            label="📥 Excel Raporunu İndir",
            data=excel_data,
            file_name="su_tuketim_analiz_raporu.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# Footer
st.markdown("---")
st.markdown("💧 Su Tüketim Analiz Sistemi | Profesyonel Dashboard | Streamlit")
