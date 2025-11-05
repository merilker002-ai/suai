import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
import warnings
import re
import os
import joblib
from sklearn.linear_model import LinearRegression
from io import BytesIO

warnings.filterwarnings('ignore')

# ======================================================================
# 🚀 STREAMLIT UYGULAMASI
# ======================================================================

st.set_page_config(
    page_title="Su Analiz & Tahmin Dashboard",
    page_icon="💧",
    layout="wide"
)

st.title("💧 Su Analiz ve Kayıp-Kaçak Tahmin Dashboard")

# İKİ ANA SEKME OLUŞTURUYORUZ
tab1, tab2 = st.tabs(["📊 Tüketim Davranış Analizi (Gelişmiş)", "📈 Kayıp-Kaçak Tahmin Modeli (ML)"])

# ======================================================================
# 📊 SEKME 1: TÜKETİM DAVRANIŞ ANALİZİ
# ======================================================================
with tab1:
    st.header("Tüketim Davranış Analizi ve Anomali Tespiti")
    
    @st.cache_data
    def load_and_analyze_data(uploaded_file, zone_file):
        """İki dosyadan veriyi okur ve gelişmiş analiz eder"""
        try:
            # Ana veri dosyasını oku - Excel formatında
            if uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
            st.success(f"✅ Ana veri başarıyla yüklendi: {len(df)} kayıt")
        except Exception as e:
            st.error(f"❌ Ana dosya okuma hatası: {e}")
            return None, None, None, None

        # Tarih formatını düzelt
        date_columns = ['ILK_OKUMA_TARIHI', 'OKUMA_TARIHI']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Tesisat numarası olan kayıtları filtrele
        if 'TESISAT_NO' in df.columns:
            df = df[df['TESISAT_NO'].notnull()]
        else:
            st.error("❌ TESISAT_NO sütunu bulunamadı!")
            return None, None, None, None
        
        # Zone veri dosyasını oku
        kullanici_zone_verileri = {}
        if zone_file is not None:
            try:
                if zone_file.name.endswith('.xlsx'):
                    zone_df = pd.read_excel(zone_file)
                else:
                    zone_df = pd.read_csv(zone_file)
                
                st.success(f"✅ Zone veri dosyası başarıyla yüklendi: {len(zone_df)} kayıt")
                
                # Zone verilerini işle - daha esnek sütun eşleştirme
                karne_col = None
                verilen_su_col = None
                tahakkuk_col = None
                kayip_oran_col = None
                
                # Sütunları bul
                for col in zone_df.columns:
                    if 'KARNE' in col.upper():
                        karne_col = col
                    elif 'VERİLEN' in col.upper() or 'SU MİKTARI' in col.upper():
                        verilen_su_col = col
                    elif 'TAHAKKUK' in col.upper():
                        tahakkuk_col = col
                    elif 'KAYIP' in col.upper() or 'KAÇAK' in col.upper():
                        kayip_oran_col = col
                
                if karne_col:
                    for idx, row in zone_df.iterrows():
                        karne_adi = str(row[karne_col]).strip()
                        
                        # Karne numarasını çıkar (ilk 4 rakam)
                        karne_no_match = re.search(r'(\d{4})', karne_adi)
                        if karne_no_match:
                            karne_no = karne_no_match.group(1)
                            
                            zone_bilgisi = {
                                'ad': karne_adi,
                                'verilen_su': row.get(verilen_su_col, 0) if verilen_su_col else 0,
                                'tahakkuk_m3': row.get(tahakkuk_col, 0) if tahakkuk_col else 0,
                                'kayip_oran': row.get(kayip_oran_col, 0) if kayip_oran_col else 0
                            }
                            
                            kullanici_zone_verileri[karne_no] = zone_bilgisi
                else:
                    st.warning("Zone dosyasında karne bilgisi bulunamadı")
                        
            except Exception as e:
                st.error(f"❌ Zone veri dosyası yüklenirken hata: {e}")

        # Davranış analizi fonksiyonları
        def perform_behavior_analysis(df):
            son_okumalar = df.sort_values('OKUMA_TARIHI').groupby('TESISAT_NO').last().reset_index()
            
            # Okuma periyodu hesapla
            if 'ILK_OKUMA_TARIHI' in son_okumalar.columns and 'OKUMA_TARIHI' in son_okumalar.columns:
                son_okumalar['OKUMA_PERIYODU_GUN'] = (son_okumalar['OKUMA_TARIHI'] - son_okumalar['ILK_OKUMA_TARIHI']).dt.days
                son_okumalar['OKUMA_PERIYODU_GUN'] = son_okumalar['OKUMA_PERIYODU_GUN'].clip(lower=1, upper=365)
            else:
                son_okumalar['OKUMA_PERIYODU_GUN'] = 30  # Varsayılan değer
            
            # Günlük tüketim hesapla
            if 'AKTIF_m3' in son_okumalar.columns:
                son_okumalar['GUNLUK_ORT_TUKETIM_m3'] = son_okumalar['AKTIF_m3'] / son_okumalar['OKUMA_PERIYODU_GUN']
                son_okumalar['GUNLUK_ORT_TUKETIM_m3'] = son_okumalar['GUNLUK_ORT_TUKETIM_m3'].clip(lower=0.001, upper=100)
            
            return son_okumalar

        def tesisat_davranis_analizi(tesisat_no, son_okuma_row, df):
            tesisat_verisi = df[df['TESISAT_NO'] == tesisat_no].sort_values('OKUMA_TARIHI')

            if len(tesisat_verisi) < 3:
                return "Yetersiz veri", "Yetersiz kayıt", "Orta"

            if 'AKTIF_m3' not in tesisat_verisi.columns:
                return "Tüketim verisi yok", "Veri eksik", "Orta"

            tuketimler = tesisat_verisi['AKTIF_m3'].values
            tarihler_series = tesisat_verisi['OKUMA_TARIHI']

            # Gelişmiş analiz
            sifir_sayisi = sum(tuketimler == 0)
            sifir_orani = sifir_sayisi / len(tuketimler)

            std_dev = np.std(tuketimler) if len(tuketimler) > 1 else 0
            mean_tuketim = np.mean(tuketimler) if len(tuketimler) > 0 else 0
            varyasyon_katsayisi = std_dev / mean_tuketim if mean_tuketim > 0 else 0

            # Risk puanı hesapla
            risk_puan = 0
            suphe_aciklamasi = ""
            suphe_donemleri = []

            # 1. Sıfır tüketim analizi
            if sifir_sayisi >= 2:
                risk_puan += 3
                suphe_aciklamasi += "Düzensiz sıfır tüketim paterni. "
                sifir_indisler = np.where(tuketimler == 0)[0]
                for idx in sifir_indisler:
                    if idx < len(tarihler_series):
                        tarih_obj = pd.Timestamp(tarihler_series.iloc[idx])
                        suphe_donemleri.append(tarih_obj.strftime('%m/%Y'))

            # 2. Yüksek varyasyon
            if varyasyon_katsayisi > 1.5:
                risk_puan += 2
                suphe_aciklamasi += "Tüketimde yüksek dalgalanma. "
            elif varyasyon_katsayisi > 1.0:
                risk_puan += 1

            # 3. Son dönem sıfır tüketim
            if len(tuketimler) > 0 and tuketimler[-1] == 0:
                risk_puan += 2
                suphe_aciklamasi += "Son dönem sıfır tüketim. "

            # 4. Anormal yüksek tüketim
            if mean_tuketim > 50:
                risk_puan += 2
                suphe_aciklamasi += "Anormal yüksek tüketim. "
            elif mean_tuketim > 20:
                risk_puan += 1

            # Risk seviyesini belirle
            if risk_puan >= 5:
                risk_seviyesi = "Yüksek"
            elif risk_puan >= 3:
                risk_seviyesi = "Orta"
            else:
                risk_seviyesi = "Düşük"

            # Yorum belirle
            if risk_seviyesi == "Düşük":
                davranis_yorumu = "Normal tüketim paterni"
            elif risk_seviyesi == "Orta":
                davranis_yorumu = "Tüketimde hafif değişiklikler"
            else:
                davranis_yorumu = "Ciddi değişiklikler gözlemleniyor"

            return davranis_yorumu, ", ".join(suphe_donemleri) if suphe_donemleri else "Yok", risk_seviyesi

        # Analiz akışı
        st.info("İlk analiz yapılıyor...")
        son_okumalar = perform_behavior_analysis(df)
        
        st.info("🔍 Gelişmiş davranış analizi yapılıyor...")
        progress_bar = st.progress(0)
        davranis_sonuclari = []
        total_tesisat = len(son_okumalar)
        
        for i, (idx, row) in enumerate(son_okumalar.iterrows()):
            yorum, supheli_donemler, risk = tesisat_davranis_analizi(row['TESISAT_NO'], row, df)
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

        # Zone analizi
        zone_analizi = None
        if 'KARNE_NO' in df.columns:
            son_tarih = df['OKUMA_TARIHI'].max() if 'OKUMA_TARIHI' in df.columns else datetime.now()
            uc_ay_once = son_tarih - timedelta(days=90)
            
            if 'OKUMA_TARIHI' in df.columns:
                son_uc_ay_df = df[df['OKUMA_TARIHI'] >= uc_ay_once]
            else:
                son_uc_ay_df = df.copy()
            
            zone_analizi = son_uc_ay_df.groupby('KARNE_NO').agg({
                'TESISAT_NO': 'count',
                'AKTIF_m3': 'sum',
                'TOPLAM_TUTAR': 'sum'
            }).reset_index()
            zone_analizi.columns = ['KARNE_NO', 'TESISAT_SAYISI', 'TOPLAM_TUKETIM', 'TOPLAM_GELIR']

            # Risk analizi
            son_uc_ay_risk = son_uc_ay_df.merge(son_okumalar[['TESISAT_NO', 'RISK_SEVIYESI']], on='TESISAT_NO', how='left')
            zone_risk_analizi = son_uc_ay_risk.groupby('KARNE_NO').agg({
                'RISK_SEVIYESI': lambda x: (x == 'Yüksek').sum(),
                'TESISAT_NO': 'count'
            }).reset_index()
            zone_risk_analizi.columns = ['KARNE_NO', 'YUKSEK_RISKLI_TESISAT', 'TOPLAM_TESISAT']
            
            zone_analizi = zone_analizi.merge(zone_risk_analizi[['KARNE_NO', 'YUKSEK_RISKLI_TESISAT']], on='KARNE_NO', how='left')
            zone_analizi['YUKSEK_RISK_ORANI'] = (zone_analizi['YUKSEK_RISKLI_TESISAT'] / zone_analizi['TESISAT_SAYISI']) * 100
            zone_analizi['YUKSEK_RISK_ORANI'] = zone_analizi['YUKSEK_RISK_ORANI'].fillna(0)

            # Kullanıcı zone verilerini birleştir
            if kullanici_zone_verileri:
                zone_analizi['KARNE_NO'] = zone_analizi['KARNE_NO'].astype(str)
                kullanici_df = pd.DataFrame.from_dict(kullanici_zone_verileri, orient='index').reset_index()
                kullanici_df = kullanici_df.rename(columns={'index': 'KARNE_NO'})
                zone_analizi = zone_analizi.merge(kullanici_df, on='KARNE_NO', how='left')

        return df, son_okumalar, zone_analizi, kullanici_zone_verileri

    # Sidebar - Dosya Yükleme
    st.sidebar.header("📁 Veri Yükleme (Analiz için)")
    
    uploaded_file_tab1 = st.sidebar.file_uploader(
        "Ana Excel/CSV dosyasını seçin",
        type=["xlsx", "csv"],
        help="Su tüketim verilerini içeren Excel veya CSV dosyasını yükleyin",
        key="tab1_main_file"
    )
    
    zone_file_tab1 = st.sidebar.file_uploader(
        "Zone Excel/CSV dosyasını seçin", 
        type=["xlsx", "csv"],
        help="Zone bilgilerini içeren Excel veya CSV dosyasını yükleyin",
        key="tab1_zone_file"
    )

    # Demo butonu
    if st.sidebar.button("🎮 Demo Modunda Çalıştır (Analiz)"):
        st.info("Demo modu aktif! Gelişmiş analiz ile çalışılıyor...")
        np.random.seed(42)
        
        # Demo verisi oluştur
        demo_data = []
        for i in range(500):
            demo_data.append({
                'TESISAT_NO': f"TS{1000 + i}",
                'AKTIF_m3': np.random.gamma(2, 10),
                'TOPLAM_TUTAR': np.random.gamma(2, 10) * 15,
                'ILK_OKUMA_TARIHI': pd.Timestamp('2023-01-01'),
                'OKUMA_TARIHI': pd.Timestamp('2024-10-31'),
                'KARNE_NO': f"80{np.random.randint(50, 71)}"
            })
        
        df = pd.DataFrame(demo_data)
        son_okumalar = df.copy()
        son_okumalar['RISK_SEVIYESI'] = np.random.choice(['Düşük', 'Orta', 'Yüksek'], size=len(son_okumalar), p=[0.7, 0.2, 0.1])
        son_okumalar['DAVRANIS_YORUMU'] = "Demo verisi"
        son_okumalar['SUPHELI_DONEMLER'] = "Yok"
        
        st.success("✅ Demo verisi başarıyla oluşturuldu!")

    # Ana akış
    if uploaded_file_tab1 is not None:
        df, son_okumalar, zone_analizi, kullanici_zone_verileri = load_and_analyze_data(uploaded_file_tab1, zone_file_tab1)
        
        if df is not None and son_okumalar is not None:
            st.success("Veri başarıyla yüklendi ve analiz edildi!")
            
            # Metrikler
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Toplam Tesisat", len(son_okumalar))
            with col2:
                st.metric("Toplam Tüketim", f"{son_okumalar['AKTIF_m3'].sum():,.0f} m³")
            with col3:
                st.metric("Toplam Gelir", f"{son_okumalar['TOPLAM_TUTAR'].sum():,.0f} TL")
            with col4:
                yuksek_risk = len(son_okumalar[son_okumalar['RISK_SEVIYESI'] == 'Yüksek'])
                st.metric("Yüksek Riskli", yuksek_risk)
            
            # Risk dağılımı
            st.subheader("Risk Seviyesi Dağılımı")
            if 'RISK_SEVIYESI' in son_okumalar.columns:
                risk_counts = son_okumalar['RISK_SEVIYESI'].value_counts()
                fig = px.pie(risk_counts, values=risk_counts.values, names=risk_counts.index, 
                           title="Risk Seviyesine Göre Tesisat Dağılımı")
                st.plotly_chart(fig, use_container_width=True)
            
            # Yüksek riskli tesisatlar
            st.subheader("Yüksek Riskli Tesisatlar")
            high_risk_data = son_okumalar[son_okumalar['RISK_SEVIYESI'] == 'Yüksek']
            st.dataframe(high_risk_data[['TESISAT_NO', 'AKTIF_m3', 'TOPLAM_TUTAR', 'DAVRANIS_YORUMU']].head(20))
            
    else:
        st.warning("⚠️ Lütfen 'Tüketim Davranış Analizi' için en az ana dosyayı yükleyin veya Demo modunu kullanın")

# ======================================================================
# 📈 SEKME 2: KAYIP-KAÇAK TAHMİN MODELİ (ML)
# ======================================================================
with tab2:
    st.header("Gelecek Dönem Kayıp-Kaçak Tahmini (Makine Öğrenimi)")
    
    MODEL_FILE = 'model.joblib'

    @st.cache_resource
    def load_model():
        """Modeli yükler"""
        if os.path.exists(MODEL_FILE):
            try:
                model = joblib.load(MODEL_FILE)
                return model
            except Exception as e:
                st.error(f"Model yüklenirken hata: {e}")
        return None

    def train_model(df):
        """Yeni model eğitir"""
        try:
            # Sütunları bul
            verilen_su_col = None
            tahakkuk_col = None
            
            for col in df.columns:
                if 'VERİLEN' in col.upper() or 'SU MİKTARI' in col.upper():
                    verilen_su_col = col
                elif 'TAHAKKUK' in col.upper():
                    tahakkuk_col = col
            
            if not verilen_su_col or not tahakkuk_col:
                st.error("Gerekli sütunlar bulunamadı!")
                return None, 0
            
            # Veriyi hazırla
            df[verilen_su_col] = pd.to_numeric(df[verilen_su_col], errors='coerce')
            df[tahakkuk_col] = pd.to_numeric(df[tahakkuk_col], errors='coerce')
            
            df_clean = df[[verilen_su_col, tahakkuk_col]].dropna()
            
            if len(df_clean) < 2:
                st.error("Yeterli veri yok!")
                return None, 0
            
            X = df_clean[[verilen_su_col]]
            y = df_clean[tahakkuk_col]
            
            model = LinearRegression()
            model.fit(X, y)
            score = model.score(X, y)
            
            return model, score
            
        except Exception as e:
            st.error(f"Model eğitme hatası: {e}")
            return None, 0

    # Model yükleme
    model = load_model()
    
    if model:
        st.success("✅ Model yüklendi!")
        
        # Tahmin arayüzü
        st.subheader("Tahmin Yap")
        future_distributed = st.number_input("Dağıtılan Su (m³):", min_value=0.0, value=10000.0, step=1000.0)
        
        if st.button("Tahmin Et"):
            predicted = model.predict([[future_distributed]])[0]
            kayip = future_distributed - predicted
            kayip_orani = (kayip / future_distributed) * 100 if future_distributed > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Tahmini Faturalanan", f"{predicted:,.0f} m³")
            col2.metric("Tahmini Kayıp", f"{kayip:,.0f} m³")
            col3.metric("Kayıp Oranı", f"%{kayip_orani:.1f}")
    
    # Model eğitme
    st.subheader("Model Eğitme")
    training_file = st.file_uploader("Eğitim verisini yükleyin", type=["xlsx", "csv"])
    
    if training_file:
        try:
            if training_file.name.endswith('.xlsx'):
                train_df = pd.read_excel(training_file)
            else:
                train_df = pd.read_csv(training_file)
            
            st.write("Veri önizleme:", train_df.head())
            
            if st.button("Modeli Eğit"):
                new_model, score = train_model(train_df)
                if new_model:
                    st.success(f"Model eğitildi! R² Skoru: {score:.3f}")
                    
                    # Modeli kaydet
                    model_bytes = BytesIO()
                    joblib.dump(new_model, model_bytes)
                    model_bytes.seek(0)
                    
                    st.download_button(
                        "Modeli İndir",
                        data=model_bytes,
                        file_name="model.joblib",
                        mime="application/octet-stream"
                    )
                    
        except Exception as e:
            st.error(f"Eğitim hatası: {e}")

# Footer
st.markdown("---")
st.markdown("💧 Su Analiz Sistemi | Streamlit Dashboard")
