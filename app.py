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

# ======================================================================
# 📊 VERİ İŞLEME FONKSİYONLARI (İKİ DOSYA OKUYAN)
# ======================================================================

@st.cache_data
def load_and_analyze_data(uploaded_file, zone_file):
    """İki dosyadan veriyi okur ve analiz eder"""
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

    # Davranış analizi fonksiyonları
    def perform_behavior_analysis(df):
        son_okumalar = df.sort_values('OKUMA_TARIHI').groupby('TESISAT_NO').last().reset_index()
        son_okumalar['OKUMA_PERIYODU_GUN'] = (son_okumalar['OKUMA_TARIHI'] - son_okumalar['ILK_OKUMA_TARIHI']).dt.days
        son_okumalar['OKUMA_PERIYODU_GUN'] = son_okumalar['OKUMA_PERIYODU_GUN'].clip(lower=1, upper=365)
        son_okumalar['GUNLUK_ORT_TUKETIM_m3'] = son_okumalar['AKTIF_m3'] / son_okumalar['OKUMA_PERIYODU_GUN']
        son_okumalar['GUNLUK_ORT_TUKETIM_m3'] = son_okumalar['GUNLUK_ORT_TUKETIM_m3'].clip(lower=0.001, upper=100)
        return son_okumalar

    son_okumalar = perform_behavior_analysis(df)
    
    # Davranış analizi fonksiyonu (şüpheli tesisat tespiti)
    def tesisat_davranis_analizi(tesisat_no, son_okuma_row, df):
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

    # Tüm tesisatlar için davranış analizi yap
    davranis_sonuclari = []
    for i, (idx, row) in enumerate(son_okumalar.iterrows()):
        yorum, supheli_donemler, risk = tesisat_davranis_analizi(row['TESISAT_NO'], row, df)
        davranis_sonuclari.append({
            'TESISAT_NO': row['TESISAT_NO'],
            'DAVRANIS_YORUMU': yorum,
            'SUPHELI_DONEMLER': supheli_donemler,
            'RISK_SEVIYESI': risk
        })

    davranis_df = pd.DataFrame(davranis_sonuclari)
    son_okumalar = son_okumalar.merge(davranis_df, on='TESISAT_NO', how='left')

    # Zone analizi
    zone_analizi = None
    if 'KARNE_NO' in df.columns:
        ekim_2024_df = df[(df['OKUMA_TARIHI'].dt.month == 10) & (df['OKUMA_TARIHI'].dt.year == 2024)]
        if len(ekim_2024_df) == 0:
            ekim_2024_df = df.copy()
        
        zone_analizi = ekim_2024_df.groupby('KARNE_NO').agg({
            'TESISAT_NO': 'count',
            'AKTIF_m3': 'sum',
            'TOPLAM_TUTAR': 'sum'
        }).reset_index()
        zone_analizi.columns = ['KARNE_NO', 'TESISAT_SAYISI', 'TOPLAM_TUKETIM', 'TOPLAM_GELIR']

        # Zone risk analizi
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

# Başlık
st.title("💧 Su Tüketim Davranış Analiz Dashboard")

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
    # Demo verisi oluştur
    st.info("Demo modu aktif! Örnek verilerle çalışılıyor...")
    np.random.seed(42)
    
    # Örnek veri oluştur
    demo_data = []
    for i in range(1000):
        tesisat_no = f"TS{1000 + i}"
        aktif_m3 = np.random.gamma(2, 10)
        toplam_tutar = aktif_m3 * 15 + np.random.normal(0, 10)
        
        demo_data.append({
            'TESISAT_NO': tesisat_no,
            'AKTIF_m3': max(aktif_m3, 0.1),
            'TOPLAM_TUTAR': max(toplam_tutar, 0),
            'ILK_OKUMA_TARIHI': pd.Timestamp('2023-01-01'),
            'OKUMA_TARIHI': pd.Timestamp('2024-10-31'),
            'KARNE_NO': f"ZONE{np.random.randint(1, 6)}"
        })
    
    df = pd.DataFrame(demo_data)
    
    # Davranış analizi
    def perform_behavior_analysis(df):
        son_okumalar = df.copy()
        son_okumalar['OKUMA_PERIYODU_GUN'] = 300
        son_okumalar['GUNLUK_ORT_TUKETIM_m3'] = son_okumalar['AKTIF_m3'] / son_okumalar['OKUMA_PERIYODU_GUN']
        return son_okumalar

    son_okumalar = perform_behavior_analysis(df)
    
    # Demo davranış analizi sonuçları
    risk_dagilimi = np.random.choice(['Düşük', 'Orta', 'Yüksek'], size=len(son_okumalar), p=[0.7, 0.2, 0.1])
    son_okumalar['RISK_SEVIYESI'] = risk_dagilimi
    son_okumalar['DAVRANIS_YORUMU'] = "Demo verisi - analiz edildi"
    son_okumalar['SUPHELI_DONEMLER'] = "Yok"
    
    # Zone analizi
    zone_analizi = df.groupby('KARNE_NO').agg({
        'TESISAT_NO': 'count',
        'AKTIF_m3': 'sum',
        'TOPLAM_TUTAR': 'sum'
    }).reset_index()
    zone_analizi.columns = ['KARNE_NO', 'TESISAT_SAYISI', 'TOPLAM_TUKETIM', 'TOPLAM_GELIR']
    
    # Örnek zone verileri
    kullanici_zone_verileri = {
        'ZONE1': {'ad': 'ÖLÇÜM NOKTASI-1 (KIRMIZI)', 'verilen_su': 20078.00, 'tahakkuk_m3': 7010.00, 'kayip_oran': 65.09},
        'ZONE2': {'ad': 'ÖLÇÜM NOKTASI-2 (MAVİ)', 'verilen_su': 3968.00, 'tahakkuk_m3': 1813.00, 'kayip_oran': 54.31},
        'ZONE3': {'ad': 'ÖLÇÜM NOKTASI-3 (ALT BÖLGE) (YEŞİL)', 'verilen_su': 19623.00, 'tahakkuk_m3': 7375.00, 'kayip_oran': 62.42},
        'ZONE4': {'ad': 'ÖLÇÜM NOKTASI-5 (ÜST BÖLGE) (MOR)', 'verilen_su': 18666.00, 'tahakkuk_m3': 7654.00, 'kayip_oran': 58.99},
        'ZONE5': {'ad': 'HASTANE BÖLGESİ (SARI)', 'verilen_su': 17775.00, 'tahakkuk_m3': 2134.00, 'kayip_oran': 87.99}
    }
    
    st.success("✅ Demo verisi başarıyla oluşturuldu!")

elif uploaded_file is not None:
    # Gerçek dosya yüklendi
    df, son_okumalar, zone_analizi, kullanici_zone_verileri = load_and_analyze_data(uploaded_file, zone_file)
else:
    st.warning("⚠️ Lütfen Excel dosyalarını yükleyin veya Demo modunu kullanın")
    st.stop()

# Genel Metrikler
if son_okumalar is not None:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Toplam Tesisat",
            value=f"{len(son_okumalar):,}"
        )
    
    with col2:
        st.metric(
            label="💧 Toplam Tüketim",
            value=f"{son_okumalar['AKTIF_m3'].sum():,.0f} m³"
        )
    
    with col3:
        st.metric(
            label="💰 Toplam Gelir",
            value=f"{son_okumalar['TOPLAM_TUTAR'].sum():,.0f} TL"
        )
    
    with col4:
        # Risk dağılımı
        yuksek_riskli = len(son_okumalar[son_okumalar['RISK_SEVIYESI'] == 'Yüksek'])
        st.metric(
            label="🚨 Yüksek Riskli Tesisat",
            value=f"{yuksek_riskli}"
        )

# Tab Menü
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Genel Görünüm", 
    "🗺️ Zone Analizi", 
    "🔍 Detaylı Analiz", 
    "📊 İleri Analiz",
    "🔥 Ateş Böceği Görünümü",
    "🔄 Boru Hattı Kaçak Tahmini"
])

with tab1:
    if son_okumalar is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            # Tüketim Dağılım Grafiği
            fig1 = px.histogram(son_okumalar, x='GUNLUK_ORT_TUKETIM_m3', 
                              title='Günlük Tüketim Dağılımı',
                              labels={'GUNLUK_ORT_TUKETIM_m3': 'Günlük Tüketim (m³)'},
                              color_discrete_sequence=['#3498DB'])
            fig1.update_layout(showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Tüketim-Tutar İlişkisi (Risk Renkli)
            fig2 = px.scatter(son_okumalar, x='AKTIF_m3', y='TOPLAM_TUTAR',
                            color='RISK_SEVIYESI',
                            title='Tüketim-Tutar İlişkisi (Risk Seviyeli)',
                            labels={'AKTIF_m3': 'Tüketim (m³)', 'TOPLAM_TUTAR': 'Toplam Tutar (TL)'},
                            color_discrete_map={'Düşük': 'green', 'Orta': 'orange', 'Yüksek': 'red'})
            st.plotly_chart(fig2, use_container_width=True)
        
        # Zaman Serisi Grafiği
        if df is not None:
            df_aylik = df.groupby(df['OKUMA_TARIHI'].dt.to_period('M')).agg({
                'AKTIF_m3': 'sum',
                'TOPLAM_TUTAR': 'sum'
            }).reset_index()
            df_aylik['OKUMA_TARIHI'] = df_aylik['OKUMA_TARIHI'].dt.to_timestamp()

            fig3 = make_subplots(specs=[[{"secondary_y": True}]])
            fig3.add_trace(
                go.Scatter(x=df_aylik['OKUMA_TARIHI'], y=df_aylik['AKTIF_m3'], 
                          name="Tüketim (m³)", line=dict(color='blue')),
                secondary_y=False,
            )
            fig3.add_trace(
                go.Scatter(x=df_aylik['OKUMA_TARIHI'], y=df_aylik['TOPLAM_TUTAR'], 
                          name="Gelir (TL)", line=dict(color='green')),
                secondary_y=True,
            )
            fig3.update_layout(title_text="Aylık Tüketim ve Gelir Trendi")
            fig3.update_xaxes(title_text="Tarih")
            fig3.update_yaxes(title_text="Tüketim (m³)", secondary_y=False)
            fig3.update_yaxes(title_text="Gelir (TL)", secondary_y=True)
            st.plotly_chart(fig3, use_container_width=True)

with tab2:
    if zone_analizi is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            # Zone Tüketim Dağılımı
            fig4 = px.pie(zone_analizi, values='TOPLAM_TUKETIM', names='KARNE_NO',
                        title='Zone Bazlı Tüketim Dağılımı')
            st.plotly_chart(fig4, use_container_width=True)
        
        with col2:
            # Zone Tesisat Sayısı
            fig5 = px.bar(zone_analizi, x='KARNE_NO', y='TESISAT_SAYISI',
                        title='Zone Bazlı Tesisat Sayısı',
                        labels={'KARNE_NO': 'Zone', 'TESISAT_SAYISI': 'Tesisat Sayısı'},
                        color_discrete_sequence=['#E74C3C'])
            st.plotly_chart(fig5, use_container_width=True)
        
        # Zone Karşılaştırma Tablosu
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
    if son_okumalar is not None:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Filtreleme Seçenekleri")
            
            # Tüketim Slider
            tuketim_range = st.slider(
                "Tüketim Aralığı (m³)",
                min_value=0,
                max_value=int(son_okumalar['AKTIF_m3'].max()) if len(son_okumalar) > 0 else 100,
                value=[0, 100],
                help="Tüketim değerine göre filtreleme yapın"
            )
            
            # Risk Seviyesi Filtresi
            risk_seviyeleri = st.multiselect(
                "Risk Seviyeleri",
                options=['Düşük', 'Orta', 'Yüksek'],
                default=['Yüksek', 'Orta']
            )
            
            # Sıralama Seçeneği
            siralama = st.selectbox(
                "Sıralama Türü",
                options=['En Yüksek Tüketim', 'En Düşük Tüketim', 'En Yüksek Risk'],
                index=2
            )
        
        with col2:
            st.subheader("Tesisat Tablosu")
            
            # Filtreleme
            min_tuketim, max_tuketim = tuketim_range
            filtreli_veri = son_okumalar[
                (son_okumalar['AKTIF_m3'] >= min_tuketim) & 
                (son_okumalar['AKTIF_m3'] <= max_tuketim) &
                (son_okumalar['RISK_SEVIYESI'].isin(risk_seviyeleri))
            ]
            
            # Sıralama
            if siralama == 'En Yüksek Tüketim':
                gosterilecek_veri = filtreli_veri.nlargest(20, 'AKTIF_m3')
            elif siralama == 'En Düşük Tüketim':
                gosterilecek_veri = filtreli_veri.nsmallest(20, 'AKTIF_m3')
            else:
                # Risk önceliğine göre sırala
                risk_sirasi = {'Yüksek': 3, 'Orta': 2, 'Düşük': 1}
                filtreli_veri['RISK_SIRASI'] = filtreli_veri['RISK_SEVIYESI'].map(risk_sirasi)
                gosterilecek_veri = filtreli_veri.nlargest(20, ['RISK_SIRASI', 'AKTIF_m3'])
            
            # Tablo gösterimi
            st.dataframe(
                gosterilecek_veri[['TESISAT_NO', 'AKTIF_m3', 'TOPLAM_TUTAR', 'GUNLUK_ORT_TUKETIM_m3', 'RISK_SEVIYESI', 'DAVRANIS_YORUMU']].round(3),
                use_container_width=True
            )

with tab4:
    if son_okumalar is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk Dağılımı
            risk_dagilim = son_okumalar['RISK_SEVIYESI'].value_counts()
            fig6 = px.pie(values=risk_dagilim.values, names=risk_dagilim.index,
                         title='Risk Seviyeleri Dağılımı',
                         color_discrete_map={'Düşük': 'green', 'Orta': 'orange', 'Yüksek': 'red'})
            st.plotly_chart(fig6, use_container_width=True)
        
        with col2:
            # Korelasyon Matrisi
            numeric_cols = son_okumalar.select_dtypes(include=[np.number]).columns
            corr_matrix = son_okumalar[numeric_cols].corr()
            
            fig7 = px.imshow(corr_matrix, 
                           title='Korelasyon Matrisi',
                           color_continuous_scale='RdBu_r',
                           aspect="auto")
            st.plotly_chart(fig7, use_container_width=True)
        
        # Aykırı Değer Analizi
        fig8 = px.box(son_okumalar, y='AKTIF_m3', 
                     title='Tüketim Dağılımı - Aykırı Değer Analizi',
                     color_discrete_sequence=['#F39C12'])
        st.plotly_chart(fig8, use_container_width=True)

with tab5:
    st.header("🔥 Ateş Böceği Görünümü - Şüpheli Tesisatlar")
    
    if son_okumalar is not None:
        # Yüksek riskli tesisatları filtrele
        yuksek_riskli = son_okumalar[son_okumalar['RISK_SEVIYESI'] == 'Yüksek']
        
        if len(yuksek_riskli) > 0:
            st.success(f"🚨 {len(yuksek_riskli)} adet yüksek riskli tesisat tespit edildi!")
            
            # Ateş böceği efekti için özel scatter plot
            fig9 = px.scatter(yuksek_riskli, x='AKTIF_m3', y='TOPLAM_TUTAR',
                            size='GUNLUK_ORT_TUKETIM_m3',
                            color='GUNLUK_ORT_TUKETIM_m3',
                            hover_name='TESISAT_NO',
                            title='🔥 Ateş Böceği Görünümü - Yüksek Riskli Tesisatlar',
                            labels={'AKTIF_m3': 'Tüketim (m³)', 'TOPLAM_TUTAR': 'Toplam Tutar (TL)'},
                            color_continuous_scale='reds',
                            size_max=30)
            
            # Ateş böceği efekti için animasyon
            fig9.update_traces(marker=dict(symbol='star', line=dict(width=2, color='DarkOrange')),
                             selector=dict(mode='markers'))
            
            st.plotly_chart(fig9, use_container_width=True)
            
            # Detaylı liste
            st.subheader("Yüksek Riskli Tesisat Detayları")
            for idx, row in yuksek_riskli.iterrows():
                with st.expander(f"🚨 Tesisat No: {row['TESISAT_NO']} - {row['DAVRANIS_YORUMU']}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Tüketim", f"{row['AKTIF_m3']:.1f} m³")
                    with col2:
                        st.metric("Tutar", f"{row['TOPLAM_TUTAR']:.1f} TL")
                    with col3:
                        st.metric("Günlük Ort.", f"{row['GUNLUK_ORT_TUKETIM_m3']:.3f} m³")
                    
                    st.write(f"**Şüpheli Dönemler:** {row['SUPHELI_DONEMLER']}")
                    st.write(f"**Davranış Yorumu:** {row['DAVRANIS_YORUMU']}")
        else:
            st.info("🎉 Hiç yüksek riskli tesisat bulunamadı!")
        
        # Orta riskli tesisatlar
        orta_riskli = son_okumalar[son_okumalar['RISK_SEVIYESI'] == 'Orta']
        if len(orta_riskli) > 0:
            st.subheader(f"🟡 Orta Riskli Tesisatlar ({len(orta_riskli)} adet)")
            
            # Orta riskliler için farklı bir görselleştirme
            fig10 = px.scatter(orta_riskli, x='AKTIF_m3', y='TOPLAM_TUTAR',
                             color='GUNLUK_ORT_TUKETIM_m3',
                             hover_name='TESISAT_NO',
                             title='Orta Riskli Tesisatlar',
                             color_continuous_scale='oranges')
            
            st.plotly_chart(fig10, use_container_width=True)

with tab6:
    st.header("🔄 Boru Hattı Kaçak Tahmini")
    
    if zone_analizi is not None and kullanici_zone_verileri:
        st.info("🔍 Zone bazlı boru hattı analizi ve kaçak tahmini")
        
        # Zone seçimi
        selected_zone = st.selectbox(
            "Zone Seçin",
            options=list(kullanici_zone_verileri.keys()),
            format_func=lambda x: f"{x} - {kullanici_zone_verileri[x]['ad']}"
        )
        
        if selected_zone:
            zone_info = kullanici_zone_verileri[selected_zone]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Verilen Su", f"{zone_info['verilen_su']:,.0f} m³")
            with col2:
                st.metric("Tahakkuk", f"{zone_info['tahakkuk_m3']:,.0f} m³")
            with col3:
                st.metric("Kayıp Oranı", f"{zone_info['kayip_oran']:.1f}%")
            
            # Kaçak tahmini hesaplama
            verilen_su = zone_info['verilen_su']
            tahakkuk = zone_info['tahakkuk_m3']
            gercek_kayip = verilen_su - tahakkuk
            
            # Zone'daki tesisat verilerini al
            if 'KARNE_NO' in df.columns:
                zone_tesisatlari = df[df['KARNE_NO'] == selected_zone]
                zone_riskli_tesisatlar = son_okumalar[
                    (son_okumalar['TESISAT_NO'].isin(zone_tesisatlari['TESISAT_NO'])) & 
                    (son_okumalar['RISK_SEVIYESI'] == 'Yüksek')
                ]
                
                # Kaçak tahmini algoritması
                def calculate_leak_estimation(zone_info, riskli_tesisatlar, zone_tesisatlari):
                    base_leak = gercek_kayip
                    
                    # Riskli tesisatlardan kaynaklı potansiyel kaçak
                    risk_leak_estimate = 0
                    if len(riskli_tesisatlar) > 0:
                        # Yüksek riskli tesisatların ortalama tüketiminin %50'si kaçak olarak tahmin ediliyor
                        avg_high_risk_consumption = riskli_tesisatlar['AKTIF_m3'].mean() if len(riskli_tesisatlar) > 0 else 0
                        risk_leak_estimate = avg_high_risk_consumption * 0.5 * len(riskli_tesisatlar)
                    
                    # Zone'daki tüketim varyasyonundan kaynaklı kaçak
                    consumption_std = zone_tesisatlari['AKTIF_m3'].std() if len(zone_tesisatlari) > 1 else 0
                    variation_leak = consumption_std * 0.3
                    
                    # Toplam tahmini kaçak
                    total_estimated_leak = risk_leak_estimate + variation_leak
                    
                    return {
                        'gercek_kayip': base_leak,
                        'risk_based_leak': risk_leak_estimate,
                        'variation_leak': variation_leak,
                        'total_estimated_leak': total_estimated_leak,
                        'explained_leak_ratio': (total_estimated_leak / base_leak * 100) if base_leak > 0 else 0
                    }
                
                leak_analysis = calculate_leak_estimation(zone_info, zone_riskli_tesisatlar, zone_tesisatlari)
                
                # Boru hattı görselleştirmesi
                st.subheader("🔧 Boru Hattı Dağıtım Ağı")
                
                # Boru hattı çizimi
                fig11 = go.Figure()
                
                # Ana boru hattı
                fig11.add_trace(go.Scatter(
                    x=[0, 1], y=[0.5, 0.5],
                    mode='lines',
                    line=dict(width=8, color='blue'),
                    name='Ana Boru Hattı'
                ))
                
                # Zone giriş noktası
                fig11.add_trace(go.Scatter(
                    x=[0], y=[0.5],
                    mode='markers+text',
                    marker=dict(size=20, color='red'),
                    text=f"ZONE {selected_zone}",
                    textposition="top center",
                    name='Zone Girişi'
                ))
                
                # Dallar (tesisatlar)
                num_branches = min(20, len(zone_tesisatlari))  # En fazla 20 dal göster
                branch_positions = np.linspace(0.1, 0.9, num_branches)
                
                for i, (idx, tesisat) in enumerate(zone_tesisatlari.head(num_branches).iterrows()):
                    # Boru dalı
                    fig11.add_trace(go.Scatter(
                        x=[0.5, 0.8], y=[0.5, branch_positions[i]],
                        mode='lines',
                        line=dict(width=3, color='green'),
                        showlegend=False
                    ))
                    
                    # Tesisat noktası
                    risk_color = 'red' if tesisat['TESISAT_NO'] in zone_riskli_tesisatlar['TESISAT_NO'].values else 'green'
                    fig11.add_trace(go.Scatter(
                        x=[0.8], y=[branch_positions[i]],
                        mode='markers+text',
                        marker=dict(size=10, color=risk_color),
                        text=tesisat['TESISAT_NO'][-4:],  # Son 4 hane
                        textposition="middle right",
                        name=f"Tesisat {tesisat['TESISAT_NO'][-4:]}",
                        hovertemplate=f"Tesisat: {tesisat['TESISAT_NO']}<br>Tüketim: {tesisat['AKTIF_m3']:.1f} m³"
                    ))
                
                fig11.update_layout(
                    title=f"{selected_zone} Boru Hattı Dağıtım Ağı",
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    showlegend=True,
                    height=500
                )
                
                st.plotly_chart(fig11, use_container_width=True)
                
                # Kaçak analizi sonuçları
                st.subheader("📊 Kaçak Tahmini Analizi")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Gerçek Kayıp", f"{leak_analysis['gercek_kayip']:,.0f} m³")
                
                with col2:
                    st.metric("Risk Bazlı Kaçak", f"{leak_analysis['risk_based_leak']:,.0f} m³")
                
                with col3:
                    st.metric("Varyasyon Kaçağı", f"{leak_analysis['variation_leak']:,.0f} m³")
                
                with col4:
                    st.metric("Toplam Tahmini", f"{leak_analysis['total_estimated_leak']:,.0f} m³")
                
                # Açıklama oranı
                st.progress(
                    min(leak_analysis['explained_leak_ratio'] / 100, 1.0),
                    text=f"Tahmini Kaçak Açıklama Oranı: {leak_analysis['explained_leak_ratio']:.1f}%"
                )
                
                # Öneriler
                st.subheader("💡 İyileştirme Önerileri")
                
                if leak_analysis['risk_based_leak'] > leak_analysis['gercek_kayip'] * 0.3:
                    st.warning("""
                    **Yüksek Riskli Tesisat Odaklı Kaçak:** 
                    - Yüksek riskli tesisatların detaylı kontrolü önerilir
                    - Bu tesisatlarda fiziki kontrol yapılmalı
                    - Sıfır tüketim paternleri incelenmeli
                    """)
                
                if leak_analysis['variation_leak'] > leak_analysis['gercek_kayip'] * 0.2:
                    st.info("""
                    **Tüketim Varyasyonu Kaynaklı Kaçak:**
                    - Tüketim dalgalanmalarının nedenleri araştırılmalı
                    - Mevsimsel etkiler değerlendirilmeli
                    - Ani tüketim değişimleri izlenmeli
                    """)
                
                if leak_analysis['explained_leak_ratio'] < 50:
                    st.error("""
                    **Yüksek Açıklanamayan Kaçak:**
                    - Fiziki boru hattı kontrolü gerekli
                    - Ana hatlarda kaçak olabilir
                    - Ölçüm sistemleri kontrol edilmeli
                    """)
                
            else:
                st.warning("Seçilen zone'a ait tesisat verisi bulunamadı.")
    else:
        st.error("Zone verisi yüklenmedi. Lütfen zone dosyasını yükleyin.")

# Footer
st.markdown("---")
st.markdown("💧 Su Tüketim Analiz Sistemi | Streamlit Dashboard | 🔥 Ateş Böceği Görünümü | 🔄 Boru Hattı Analizi")
