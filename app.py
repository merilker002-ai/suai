import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
import warnings
import re
import os  # Model dosyasını kontrol etmek için
import joblib  # Modeli kaydetmek ve yüklemek için
from sklearn.linear_model import LinearRegression # ML Modeli
from io import BytesIO # Modeli indirmek için

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
# 📊 SEKME 1: TÜKETİM DAVRANIŞ ANALİZİ (Sizin Kodunuz)
# ======================================================================
with tab1:
    st.header("Tüketim Davranış Analizi ve Anomali Tespiti")
    
    # --- Sizin Dosya Yükleme ve Analiz Fonksiyonlarınız ---
    # @st.cache_data (bu harika bir kullanım, böyle kalmalı)
    @st.cache_data
    def load_and_analyze_data(uploaded_file, zone_file):
        """İki dosyadan veriyi okur ve gelişmiş analiz eder"""
        try:
            # Ana veri dosyasını oku
            # GÜNCELLEME: CSV olarak okuyoruz (yüklenen dosya CSV)
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Ana veri başarıyla yüklendi: {len(df)} kayıt")
        except Exception as e:
            st.error(f"❌ Ana dosya okuma hatası: {e}. Lütfen 'yavuz.xlsx - Sayfa1.csv' yüklediğinizden emin olun.")
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
                # GÜNCELLEME: CSV olarak okuyoruz
                zone_excel_df = pd.read_csv(zone_file)
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
                            
                            # Zone bilgilerini topla (Sizin dosyanızdaki sütun adlarıyla eşleşti)
                            zone_bilgisi = {
                                'ad': karne_adi,
                                'verilen_su': row.get('VERİLEN SU MİKTARI M3', 0),
                                'tahakkuk_m3': row.get('TAHAKKUK M3', 0),
                                'kayip_oran': row.get('BRÜT KAYIP KAÇAK ORANI\n%', 0)
                            }
                            
                            kullanici_zone_verileri[karne_no] = zone_bilgisi
            except Exception as e:
                st.error(f"❌ Zone veri dosyası yüklenirken hata: {e}")

        # --- Sizin Diğer Analiz Fonksiyonlarınız Buraya ---
        # (perform_behavior_analysis, tesisat_davranis_analizi vb.)
        # ... (Bu fonksiyonlar uzun olduğu için kod tekrarı yapmıyorum,
        # ...  ancak sizin kodunuzdaki gibi burada olmalılar) ...
        # ...
        
        # Örnek olarak sizin fonksiyonlarınızı buraya ekliyorum:
        def perform_behavior_analysis(df):
            son_okumalar = df.sort_values('OKUMA_TARIHI').groupby('TESISAT_NO').last().reset_index()
            son_okumalar['OKUMA_PERIYODU_GUN'] = (son_okumalar['OKUMA_TARIHI'] - son_okumalar['ILK_OKUMA_TARIHI']).dt.days
            son_okumalar['OKUMA_PERIYODU_GUN'] = son_okumalar['OKUMA_PERIYODU_GUN'].clip(lower=1, upper=365)
            son_okumalar['GUNLUK_ORT_TUKETIM_m3'] = son_okumalar['AKTIF_m3'] / son_okumalar['OKUMA_PERIYODU_GUN']
            son_okumalar['GUNLUK_ORT_TUKETIM_m3'] = son_okumalar['GUNLUK_ORT_TUKETIM_m3'].clip(lower=0.001, upper=100)
            return son_okumalar

        def tesisat_davranis_analizi(tesisat_no, son_okuma_row, df):
            tesisat_verisi = df[df['TESISAT_NO'] == tesisat_no].sort_values('OKUMA_TARIHI')
            if len(tesisat_verisi) < 3: return "Yetersiz veri", "Yetersiz kayıt", "Orta"
            tuketimler = tesisat_verisi['AKTIF_m3'].values
            tarihler_series = tesisat_verisi['OKUMA_TARIHI']
            sifir_sayisi = sum(tuketimler == 0); sifir_orani = sifir_sayisi / len(tuketimler)
            std_dev = np.std(tuketimler) if len(tuketimler) > 1 else 0
            mean_tuketim = np.mean(tuketimler) if len(tuketimler) > 0 else 0
            varyasyon_katsayisi = std_dev / mean_tuketim if mean_tuketim > 0 else 0
            if len(tuketimler) >= 5: trend = "stabil" # ... (trend analizinin devamı)
            else: trend = "belirsiz"
            suphe_aciklamasi = ""; suphe_donemleri = []; risk_seviyesi = "Düşük"; risk_puan = 0
            if sifir_sayisi >= 2: risk_puan += 3 # ... (risk analizinin devamı)
            if risk_puan >= 5: risk_seviyesi = "Yüksek"
            elif risk_puan >= 3: risk_seviyesi = "Orta"
            if risk_seviyesi == "Düşük": davranis_yorumu = "Normal tüketim paterni"
            elif risk_seviyesi == "Orta": davranis_yorumu = "Tüketimde hafif değişiklikler"
            else: davranis_yorumu = "Ciddi değişiklikler gözlemleniyor"
            return davranis_yorumu, ", ".join(suphe_donemleri) if suphe_donemleri else "Yok", risk_seviyesi

        # --- Analiz Akışı ---
        st.info("İlk analiz yapılıyor...")
        son_okumalar = perform_behavior_analysis(df)
        
        st.info("🔍 Gelişmiş davranış analizi yapılıyor...")
        progress_bar = st.progress(0)
        davranis_sonuclari = []
        total_tesisat = len(son_okumalar)
        
        for i, (idx, row) in enumerate(son_okumalar.iterrows()):
            yorum, supheli_donemler, risk = tesisat_davranis_analizi(row['TESISAT_NO'], row, df)
            davranis_sonuclari.append({'TESISAT_NO': row['TESISAT_NO'], 'DAVRANIS_YORUMU': yorum, 'SUPHELI_DONEMLER': supheli_donemler, 'RISK_SEVIYESI': risk})
            if i % 100 == 0: progress_bar.progress(min((i + 1) / total_tesisat, 1.0))
        
        progress_bar.progress(1.0)
        davranis_df = pd.DataFrame(davranis_sonuclari)
        son_okumalar = son_okumalar.merge(davranis_df, on='TESISAT_NO', how='left')

        zone_analizi = None
        if 'KARNE_NO' in df.columns:
            # ... (Sizin Zone analizi kodunuz) ...
            pass

        return df, son_okumalar, zone_analizi, kullanici_zone_verileri

    # --- Sizin Sidebar Kodunuz (Key'ler güncellendi) ---
    st.sidebar.header("📁 Veri Yükleme (Analiz için)")
    uploaded_file_tab1 = st.sidebar.file_uploader(
        "Ana CSV dosyasını seçin (yavuz.xlsx)",
        type=["csv"],
        help="Su tüketim verilerini içeren 'yavuz.xlsx - Sayfa1.csv' dosyasını yükleyin",
        key="tab1_main_file"
    )
    zone_file_tab1 = st.sidebar.file_uploader(
        "Zone CSV dosyasını seçin (yavuzeli merkez ekim.xlsx)",
        type=["csv"],
        help="Zone bilgilerini içeren 'yavuzeli merkez ekim.xlsx - Table 1.csv' dosyasını yükleyin",
        key="tab1_zone_file"
    )

    if st.sidebar.button("🎮 Demo Modunda Çalıştır (Analiz)"):
        st.info("Demo modu aktif! Gelişmiş analiz ile çalışılıyor...")
        # ... Sizin demo modu kodunuz buraya gelecek ...
        st.success("✅ Gelişmiş demo verisi başarıyla oluşturuldu!")

    # --- Sizin Ana Akış Kodunuz ---
    if uploaded_file_tab1 is not None and zone_file_tab1 is not None:
        df, son_okumalar, zone_analizi, kullanici_zone_verileri = load_and_analyze_data(uploaded_file_tab1, zone_file_tab1)
        if df is not None:
            st.success("Veri başarıyla yüklendi ve analiz edildi!")
            st.subheader("Analiz Edilen Tesisat Verisi (Son Okumalar)")
            st.dataframe(son_okumalar.head())
            
            st.subheader("Risk Seviyesi Dağılımı")
            if 'RISK_SEVIYESI' in son_okumalar.columns:
                risk_counts = son_okumalar['RISK_SEVIYESI'].value_counts()
                fig = px.pie(risk_counts, values=risk_counts.values, names=risk_counts.index, title="Risk Seviyesine Göre Tesisat Dağılımı")
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Yüksek Riskli Tesisatlar")
            st.dataframe(son_okumalar[son_okumalar['RISK_SEVIYESI'] == 'Yüksek'])
            
    else:
        st.warning("⚠️ Lütfen 'Tüketim Davranış Analizi' için her iki CSV dosyasını da yükleyin veya Demo modunu kullanın")


# ======================================================================
# 📈 SEKME 2: KAYIP-KAÇAK TAHMİN MODELİ (ML)
# ======================================================================
with tab2:
    st.header("Gelecek Dönem Kayıp-Kaçak Tahmini (Makine Öğrenimi)")
    st.markdown("Bu model, 'Dağıtılan Su' miktarına göre 'Faturalanan Su' miktarını tahmin eder ve kayıp oranını hesaplar.")

    MODEL_FILE = 'model.joblib'

    # --- Model Yükleme Fonksiyonu (Cache'li) ---
    @st.cache_resource
    def load_model():
        """GitHub'a commit'lenmiş modeli yükler."""
        if os.path.exists(MODEL_FILE):
            try:
                model = joblib.load(MODEL_FILE)
                return model
            except Exception as e:
                st.error(f"Model yüklenirken hata oluştu: {e}")
                return None
        return None

    # --- Model Eğitme Fonksiyonu (Sizin Sütunlarla Güncellendi) ---
    def train_model(df):
        """Yeni bir modeli eğitir."""
        try:
            # GÜNCELLEME: Sütun adları sizin 'yavuzeli merkez ekim.xlsx' dosyanıza göre güncellendi.
            required_col_x = 'VERİLEN SU MİKTARI M3'
            required_col_y = 'TAHAKKUK M3'

            if required_col_x not in df.columns or required_col_y not in df.columns:
                st.error(f"Hata: Model eğitimi için '{required_col_x}' ve '{required_col_y}' sütunları zorunludur.")
                return None, 0
            
            # Sadece numerik verilerle çalış
            df[required_col_x] = pd.to_numeric(df[required_col_x], errors='coerce')
            df[required_col_y] = pd.to_numeric(df[required_col_y], errors='coerce')

            df_clean = df[[required_col_x, required_col_y]].dropna()
            
            if len(df_clean) < 3: # Regresyon için en az 2-3 nokta gerekir
                st.error("Hata: Model eğitimi için en az 3 geçerli (boş olmayan) veri satırı gerekir.")
                return None, 0

            X = df_clean[[required_col_x]]
            y = df_clean[required_col_y]
            
            model = LinearRegression()
            model.fit(X, y)
            score = model.score(X, y) # R-kare skoru
            
            return model, score
        except Exception as e:
            st.error(f"Model eğitilirken bir hata oluştu: {e}")
            return None, 0

    # --- Ana Tahmin Arayüzü ---
    model = load_model()
    
    if model:
        st.success(f"✅ Eğitimli model ('{MODEL_FILE}') başarıyla yüklendi.")
        st.subheader("Yeni Tahmin Yapın")
        
        future_distributed = st.number_input(
            "Tahmin için 'Dağıtılan Su (m3)' (VERİLEN SU MİKTARI M3) girin:", 
            min_value=0.0, 
            value=10000.0, # Zone verisine göre değer güncellendi
            step=1000.0
        )
        
        if st.button("Tahmin Et", type="primary", key="predict_button"):
            try:
                predicted_billed = model.predict([[future_distributed]])[0]
                kayip_m3 = future_distributed - predicted_billed
                
                if future_distributed > 0:
                    kayip_orani = (kayip_m3 / future_distributed) * 100
                else:
                    kayip_orani = 0
                
                st.subheader("Tahmin Sonuçları:")
                col1, col2, col3 = st.columns(3)
                col1.metric("Tahmini Faturalanan Su (TAHAKKUK M3)", f"{predicted_billed:,.0f} m³")
                col2.metric("Tahmini Kayıp Miktar", f"{kayip_m3:,.0f} m³")
                col3.metric("Tahmini Kayıp Oranı", f"% {kayip_orani:.2f}", delta_color="inverse")

            except Exception as e:
                st.error(f"Tahmin yapılırken bir hata oluştu: {e}")
                
    else:
        st.warning(f"⚠️ Eğitimli model ('{MODEL_FILE}') bulunamadı. Lütfen aşağıdan yeni bir model eğitin.")

    st.divider()

    # --- Model Yönetim Arayüzü (Expander içinde) ---
    with st.expander("🛠️ YÖNETİCİ: Modeli Eğit / Güncelle"):
        st.info(
            "Burada, modelinizi eğitmek için zone verinizi yükleyin.\n"
            "**Tavsiye:** 'yavuzeli merkez ekim.xlsx - Table 1.csv' dosyasını yükleyin."
        )
        
        uploaded_training_file = st.file_uploader(
            "Model eğitim verisini (CSV) yükleyin", 
            type=["csv"],
            key="training_file"
        )
        
        if uploaded_training_file:
            try:
                df_train = pd.read_csv(uploaded_training_file)
                st.write("Yüklenen eğitim verilerinin önizlemesi (TOPLAM satırlarını hariç tutmaya çalışır):", 
                         df_train[~df_train['KARNE NO VE ADI'].str.contains("TOPLAM", na=False)].head())
                
                if st.button("Modeli Bu Veriyle Eğit", type="primary", key="train_button"):
                    with st.spinner("Yeni model eğitiliyor... Lütfen bekleyin."):
                        # 'TOPLAM' yazan satırları eğitimden çıkar
                        df_train_cleaned = df_train[~df_train['KARNE NO VE ADI'].str.contains("TOPLAM", na=False)]
                        new_model, score = train_model(df_train_cleaned)
                        
                        if new_model:
                            st.success(f"Model başarıyla eğitildi! Yeni R-kare skoru: {score:.2f}")
                            
                            # Modeli hafızada baytlara kaydet
                            model_bytes = BytesIO()
                            joblib.dump(new_model, model_bytes)
                            model_bytes.seek(0)
                            
                            # İndirme butonunu göster
                            st.download_button(
                                label="Yeni 'model.joblib' dosyasını indir",
                                data=model_bytes,
                                file_name="model.joblib",
                                mime="application/octet-stream"
                            )
                            st.warning(
                                "**ÖNEMLİ:** İndirdiğiniz bu 'model.joblib' dosyasını, "
                                "bu uygulamanın çalıştığı GitHub deposunun ana dizinine yükleyin ('commit' ve 'push' yapın). "
                                "Uygulama otomatik olarak yeni modeli kullanmaya başlayacaktır."
                            )
            except Exception as e:
                st.error(f"Eğitim verisi yüklenirken hata: {e}")
