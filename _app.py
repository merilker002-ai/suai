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
import json
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import joblib
import requests
import hashlib

warnings.filterwarnings('ignore')

# ======================================================================
# 🧠 AKILLI ÖĞRENEN YAPAY ZEKA SİSTEMİ
# ======================================================================

class AkilliSuAnalizi:
    def __init__(self):
        self.model_dosyasi = "su_analizi_modeli.pkl"
        self.ogrenme_verisi_dosyasi = "ogrenme_verisi.json"
        self.github_repo = "https://api.github.com/repos/kullanici/su-analizi-verileri"
        self.github_token = st.secrets.get("GITHUB_TOKEN", "")  # Streamlit secrets'tan al
        
    def modeli_yukle_veya_olustur(self):
        """Modeli yükle veya yeni oluştur"""
        try:
            # Önce yerel dosyadan dene
            if os.path.exists(self.model_dosyasi):
                model_verisi = joblib.load(self.model_dosyasi)
                st.sidebar.success("✅ Yerel model yüklendi")
                return model_verisi
            else:
                # GitHub'dan dene
                return self.githubdan_model_yukle()
        except:
            # Yeni model oluştur
            st.sidebar.info("🆕 Yeni model oluşturuluyor...")
            return self.yeni_model_olustur()
    
    def yeni_model_olustur(self):
        """Yeni model oluştur"""
        model_verisi = {
            'isolation_forest': IsolationForest(contamination=0.1, random_state=42),
            'risk_predictor': RandomForestRegressor(n_estimators=100, random_state=42),
            'scaler': StandardScaler(),
            'cluster': DBSCAN(eps=0.5, min_samples=5),
            'ogrenme_sayaci': 0,
            'son_ogrenme_tarihi': datetime.now().isoformat(),
            'model_version': '1.0',
            'ogrenme_verisi': []
        }
        return model_verisi
    
    def githubdan_model_yukle(self):
        """GitHub'dan model yükle"""
        try:
            if self.github_token:
                headers = {'Authorization': f'token {self.github_token}'}
                response = requests.get(f"{self.github_repo}/contents/{self.model_dosyasi}", 
                                      headers=headers)
                if response.status_code == 200:
                    content = response.json()['content']
                    # Base64 decode ve modeli yükle
                    import base64
                    model_bytes = base64.b64decode(content)
                    with open('temp_model.pkl', 'wb') as f:
                        f.write(model_bytes)
                    model_verisi = joblib.load('temp_model.pkl')
                    os.remove('temp_model.pkl')
                    st.sidebar.success("✅ GitHub'dan model yüklendi")
                    return model_verisi
        except:
            pass
        return self.yeni_model_olustur()
    
    def modeli_kaydet(self, model_verisi):
        """Modeli kaydet"""
        try:
            joblib.dump(model_verisi, self.model_dosyasi)
            
            # GitHub'a da yükle
            if self.github_token:
                self.githuba_model_yukle()
                
        except Exception as e:
            st.sidebar.warning(f"⚠️ Model kaydedilemedi: {e}")
    
    def githuba_model_yukle(self):
        """Modeli GitHub'a yükle"""
        try:
            if os.path.exists(self.model_dosyasi):
                with open(self.model_dosyasi, 'rb') as f:
                    content = f.read()
                
                # Base64 encode
                import base64
                content_b64 = base64.b64encode(content).decode()
                
                headers = {
                    'Authorization': f'token {self.github_token}',
                    'Content-Type': 'application/json'
                }
                
                # Dosya var mı kontrol et
                check_response = requests.get(
                    f"{self.github_repo}/contents/{self.model_dosyasi}",
                    headers=headers
                )
                
                data = {
                    "message": f"Model güncelleme - {datetime.now()}",
                    "content": content_b64,
                    "branch": "main"
                }
                
                if check_response.status_code == 200:
                    # Dosya varsa, SHA ekle
                    data["sha"] = check_response.json()["sha"]
                
                response = requests.put(
                    f"{self.github_repo}/contents/{self.model_dosyasi}",
                    headers=headers,
                    data=json.dumps(data)
                )
                
                if response.status_code in [200, 201]:
                    st.sidebar.success("✅ Model GitHub'a kaydedildi")
                else:
                    st.sidebar.warning("⚠️ GitHub'a yükleme başarısız")
                    
        except Exception as e:
            st.sidebar.warning(f"⚠️ GitHub yükleme hatası: {e}")
    
    def veri_ozellikleri_cikar(self, df):
        """Veriden özellikler çıkar"""
        ozellikler = []
        
        for tesisat_no in df['TESISAT_NO'].unique():
            tesisat_verisi = df[df['TESISAT_NO'] == tesisat_no]
            
            if len(tesisat_verisi) < 2:
                continue
                
            tuketimler = tesisat_verisi['AKTIF_m3'].values
            
            ozellik = {
                'tesisat_no': tesisat_no,
                'ortalama_tuketim': np.mean(tuketimler),
                'std_tuketim': np.std(tuketimler),
                'varyasyon_katsayisi': np.std(tuketimler) / np.mean(tuketimler) if np.mean(tuketimler) > 0 else 0,
                'max_tuketim': np.max(tuketimler),
                'min_tuketim': np.min(tuketimler),
                'sifir_sayisi': np.sum(tuketimler == 0),
                'son_tuketim': tuketimler[-1],
                'veri_uzunlugu': len(tuketimler),
                'trend': tuketimler[-1] - tuketimler[0] if len(tuketimler) > 1 else 0
            }
            ozellikler.append(ozellik)
        
        return pd.DataFrame(ozellikler)
    
    def modeli_egit(self, model_verisi, yeni_veri_df):
        """Modeli yeni veriyle eğit"""
        try:
            # Özellikleri çıkar
            ozellikler_df = self.veri_ozellikleri_cikar(yeni_veri_df)
            
            if len(ozellikler_df) == 0:
                return model_verisi
            
            # Sayısal özellikleri seç
            numeric_columns = ['ortalama_tuketim', 'std_tuketim', 'varyasyon_katsayisi', 
                             'max_tuketim', 'min_tuketim', 'sifir_sayisi', 'son_tuketim', 'trend']
            X = ozellikler_df[numeric_columns].fillna(0)
            
            # Scaler'ı güncelle
            if len(model_verisi['ogrenme_verisi']) > 0:
                # Mevcut veriyle birleştir
                eski_veri = pd.DataFrame(model_verisi['ogrenme_verisi'])
                X_combined = pd.concat([eski_veri[numeric_columns], X], ignore_index=True)
                model_verisi['scaler'].fit(X_combined)
            else:
                model_verisi['scaler'].fit(X)
            
            X_scaled = model_verisi['scaler'].transform(X)
            
            # Modelleri güncelle (incremental learning)
            model_verisi['isolation_forest'].fit(X_scaled)
            model_verisi['risk_predictor'].fit(X_scaled, ozellikler_df['varyasyon_katsayisi'])
            
            # Öğrenme verisini güncelle (sadece son 1000 kayıt)
            yeni_ogrenme_verisi = X.to_dict('records')
            model_verisi['ogrenme_verisi'].extend(yeni_ogrenme_verisi)
            model_verisi['ogrenme_verisi'] = model_verisi['ogrenme_verisi'][-1000:]
            
            model_verisi['ogrenme_sayaci'] += 1
            model_verisi['son_ogrenme_tarihi'] = datetime.now().isoformat()
            
            st.sidebar.success(f"🧠 Model güncellendi (Toplam öğrenme: {model_verisi['ogrenme_sayaci']})")
            
        except Exception as e:
            st.sidebar.warning(f"⚠️ Model eğitim hatası: {e}")
        
        return model_verisi
    
    def tahmin_yap(self, model_verisi, tesisat_verisi):
        """Tahmin yap"""
        try:
            ozellikler_df = self.veri_ozellikleri_cikar(tesisat_verisi)
            
            if len(ozellikler_df) == 0:
                return []
            
            numeric_columns = ['ortalama_tuketim', 'std_tuketim', 'varyasyon_katsayisi', 
                             'max_tuketim', 'min_tuketim', 'sifir_sayisi', 'son_tuketim', 'trend']
            X = ozellikler_df[numeric_columns].fillna(0)
            X_scaled = model_verisi['scaler'].transform(X)
            
            # Anomali skorları
            anomaly_scores = model_verisi['isolation_forest'].decision_function(X_scaled)
            risk_scores = model_verisi['risk_predictor'].predict(X_scaled)
            
            # Risk seviyelerini belirle
            tahminler = []
            for i, (idx, row) in enumerate(ozellikler_df.iterrows()):
                anomaly_score = anomaly_scores[i]
                risk_score = risk_scores[i]
                
                # Kombine risk skoru
                combined_risk = (1 - anomaly_score) * 0.6 + risk_score * 0.4
                
                if combined_risk > 0.7:
                    risk_seviyesi = "Yüksek"
                elif combined_risk > 0.4:
                    risk_seviyesi = "Orta"
                else:
                    risk_seviyesi = "Düşük"
                
                tahminler.append({
                    'TESISAT_NO': row['tesisat_no'],
                    'RISK_SEVIYESI': risk_seviyesi,
                    'RISK_SKORU': combined_risk,
                    'ANOMALI_SKORU': anomaly_score
                })
            
            return tahminler
            
        except Exception as e:
            st.sidebar.warning(f"⚠️ Tahmin hatası: {e}")
            return []

# ======================================================================
# 🚀 STREAMLIT UYGULAMASI
# ======================================================================

st.set_page_config(
    page_title="Akıllı Su Tüketim Analiz Dashboard",
    page_icon="🧠",
    layout="wide"
)

# Akıllı sistemi başlat
akilli_sistem = AkilliSuAnalizi()

# ======================================================================
# 📊 VERİ İŞLEME FONKSİYONLARI
# ======================================================================

@st.cache_data
def load_and_analyze_data(uploaded_file, zone_file, model_verisi):
    """İki dosyadan veriyi okur ve akıllı analiz yapar"""
    try:
        # Ana veri dosyasını oku
        df = pd.read_excel(uploaded_file)
        st.success(f"✅ Ana veri başarıyla yüklendi: {len(df)} kayıt")
    except Exception as e:
        st.error(f"❌ Ana dosya okuma hatası: {e}")
        return None, None, None, None, None

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
                if 'KARNE NO VE ADI' in row.index:
                    karne_adi = str(row['KARNE NO VE ADI']).strip()
                    
                    karne_no_match = re.search(r'(\d{4})', karne_adi)
                    if karne_no_match:
                        karne_no = karne_no_match.group(1)
                        
                        zone_bilgisi = {
                            'ad': karne_adi,
                            'verilen_su': row.get('VERİLEN SU MİKTARI M3', 0),
                            'tahakkuk_m3': row.get('TAHAKKUK M3', 0),
                            'kayip_oran': row.get('BRÜT KAYIP KAÇAK ORANI\n%', 0)
                        }
                        
                        kullanici_zone_verileri[karne_no] = zone_bilgisi
        except Exception as e:
            st.error(f"❌ Zone veri dosyası yüklenirken hata: {e}")

    # Modeli yeni veriyle eğit
    model_verisi = akilli_sistem.modeli_egit(model_verisi, df)
    
    # Akıllı tahmin yap
    akilli_tahminler = akilli_sistem.tahmin_yap(model_verisi, df)
    
    # Davranış analizi
    def perform_behavior_analysis(df):
        son_okumalar = df.sort_values('OKUMA_TARIHI').groupby('TESISAT_NO').last().reset_index()
        son_okumalar['OKUMA_PERIYODU_GUN'] = (son_okumalar['OKUMA_TARIHI'] - son_okumalar['ILK_OKUMA_TARIHI']).dt.days
        son_okumalar['OKUMA_PERIYODU_GUN'] = son_okumalar['OKUMA_PERIYODU_GUN'].clip(lower=1, upper=365)
        son_okumalar['GUNLUK_ORT_TUKETIM_m3'] = son_okumalar['AKTIF_m3'] / son_okumalar['OKUMA_PERIYODU_GUN']
        son_okumalar['GUNLUK_ORT_TUKETIM_m3'] = son_okumalar['GUNLUK_ORT_TUKETIM_m3'].clip(lower=0.001, upper=100)
        return son_okumalar

    son_okumalar = perform_behavior_analysis(df)
    
    # Akıllı tahminleri birleştir
    if akilli_tahminler:
        tahmin_df = pd.DataFrame(akilli_tahminler)
        son_okumalar = son_okumalar.merge(tahmin_df, on='TESISAT_NO', how='left')
        
        # Eksik risk seviyeleri için geleneksel yöntem
        mask = son_okumalar['RISK_SEVIYESI'].isna()
        son_okumalar.loc[mask, 'RISK_SEVIYESI'] = son_okumalar.loc[mask, 'AKTIF_m3'].apply(
            lambda x: 'Yüksek' if x > 50 else 'Orta' if x > 20 else 'Düşük'
        )
    else:
        # Geleneksel risk belirleme
        son_okumalar['RISK_SEVIYESI'] = son_okumalar['AKTIF_m3'].apply(
            lambda x: 'Yüksek' if x > 50 else 'Orta' if x > 20 else 'Düşük'
        )
        son_okumalar['RISK_SKORU'] = son_okumalar['AKTIF_m3'] / son_okumalar['AKTIF_m3'].max()
    
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

    return df, son_okumalar, zone_analizi, kullanici_zone_verileri, model_verisi

# ======================================================================
# 🎨 STREAMLIT ARAYÜZ
# ======================================================================

# Başlık
st.title("🧠 Akıllı Su Tüketim Analiz Dashboard")

# Modeli yükle
model_verisi = akilli_sistem.modeli_yukle_veya_olustur()

# Sidebar
st.sidebar.header("📁 Dosya Yükleme")

uploaded_file = st.sidebar.file_uploader(
    "Ana Excel dosyasını seçin (yavuz.xlsx)",
    type=["xlsx"],
    help="Su tüketim verilerini içeren Excel dosyasını yükleyin"
)

zone_file = st.sidebar.file_uploader(
    "Zone Excel dosyasını seçin (yavuzeli_merkez_ekim.xlsx)",
    type=["xlsx"],
    help="Zone bilgilerini içeren Excel dosyasını yükleyin"
)

# Model bilgileri
st.sidebar.header("🧠 Model Bilgileri")
st.sidebar.info(f"Öğrenme Sayacı: {model_verisi['ogrenme_sayaci']}")
st.sidebar.info(f"Son Güncelleme: {model_verisi['son_ogrenme_tarihi'][:10]}")

# Modeli kaydet butonu
if st.sidebar.button("💾 Modeli Kaydet"):
    akilli_sistem.modeli_kaydet(model_verisi)

# Demo butonu
if st.sidebar.button("🎮 Demo Modunda Çalıştır"):
    # Demo verisi oluştur
    st.info("Demo modu aktif! Örnek verilerle çalışılıyor...")
    np.random.seed(42)
    
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
            'KARNE_NO': f"80{np.random.randint(50, 71)}"
        })
    
    df = pd.DataFrame(demo_data)
    
    # Demo için basit analiz
    def perform_behavior_analysis(df):
        son_okumalar = df.copy()
        son_okumalar['OKUMA_PERIYODU_GUN'] = 300
        son_okumalar['GUNLUK_ORT_TUKETIM_m3'] = son_okumalar['AKTIF_m3'] / son_okumalar['OKUMA_PERIYODU_GUN']
        return son_okumalar

    son_okumalar = perform_behavior_analysis(df)
    
    risk_dagilimi = np.random.choice(['Düşük', 'Orta', 'Yüksek'], size=len(son_okumalar), p=[0.7, 0.2, 0.1])
    son_okumalar['RISK_SEVIYESI'] = risk_dagilimi
    son_okumalar['RISK_SKORU'] = np.random.random(len(son_okumalar))
    
    # Demo zone verileri
    kullanici_zone_verileri = {
        '8050': {'ad': 'ÖLÇÜM NOKTASI-5 (ÜST BÖLGE) (MOR)', 'verilen_su': 18666.00, 'tahakkuk_m3': 7654.00, 'kayip_oran': 58.99},
        '8055': {'ad': 'ÖLÇÜM NOKTASI-3 (ALT BÖLGE) (YEŞİL)', 'verilen_su': 19623.00, 'tahakkuk_m3': 7375.00, 'kayip_oran': 62.42},
        '8060': {'ad': 'ÖLÇÜM NOKTASI-1 (KIRMIZI)', 'verilen_su': 20078.00, 'tahakkuk_m3': 7010.00, 'kayip_oran': 65.09},
        '8065': {'ad': 'ÖLÇÜM NOKTASI-2 (MAVİ)', 'verilen_su': 3968.00, 'tahakkuk_m3': 1813.00, 'kayip_oran': 54.31},
        '8070': {'ad': 'HASTANE BÖLGESİ (SARI)', 'verilen_su': 17775.00, 'tahakkuk_m3': 2134.00, 'kayip_oran': 87.99}
    }
    
    zone_analizi = df.groupby('KARNE_NO').agg({
        'TESISAT_NO': 'count',
        'AKTIF_m3': 'sum',
        'TOPLAM_TUTAR': 'sum'
    }).reset_index()
    zone_analizi.columns = ['KARNE_NO', 'TESISAT_SAYISI', 'TOPLAM_TUKETIM', 'TOPLAM_GELIR']
    
    st.success("✅ Demo verisi başarıyla oluşturuldu!")

elif uploaded_file is not None:
    # Gerçek dosya yüklendi
    df, son_okumalar, zone_analizi, kullanici_zone_verileri, model_verisi = load_and_analyze_data(
        uploaded_file, zone_file, model_verisi
    )
else:
    st.warning("⚠️ Lütfen Excel dosyalarını yükleyin veya Demo modunu kullanın")
    st.stop()

# Geri kalan dashboard kodu aynı kalacak...
# (Önceki kodun tab1, tab2, tab3, tab4, tab5, tab6 kısımları buraya gelecek)

# Tab Menü - Kısaltılmış olarak gösteriyorum
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

# Diğer tab'lar önceki kodla aynı şekilde devam edecek...
# ... (tab2, tab3, tab4, tab5, tab6 içerikleri)

# Footer
st.markdown("---")
st.markdown("🧠 Akıllı Su Tüketim Analiz Sistemi | Öğrenen Yapay Zeka | Streamlit Dashboard")
