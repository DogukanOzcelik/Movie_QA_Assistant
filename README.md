# 🎬 TMDB Movie Q&A Assistant

## 🎥 Demo

![Uygulama Çalışma Videosu](assets/demo.gif)

Akbank Generative AI Bootcamp için hazırlanmış RAG (Retrieval-Augmented Generation) tabanlı film asistanı projesi.

## 📋 Proje Hakkında

Bu proje, TMDB 5000 Movies veri seti üzerinde film bilgisi hakkında sorular sorabileceğiniz bir AI asistanı oluşturur. Hugging Face'teki TMDB veri setini kullanarak, kullanıcıların sorularına ilgili filmlerden bilgi çekerek **İngilizce** yanıt verir. Kod içi yorumlar **Türkçe** olarak eklenmiştir.

## 🛠️ Kullanılan Teknolojiler

- **Haystack 2.x**: RAG pipeline framework
- **Streamlit**: Web arayüzü
- **Sentence Transformers**: İngilizce embedding modeli (all-mpnet-base-v2)
- **Google Gemini**: Text generation modeli (gemini-2.0-flash-exp)
- **InMemory Document Store**: Vektör veritabanı
- **Hugging Face Datasets**: Veri seti yönetimi (AiresPucrs/tmdb-5000-movies)
- **RapidFuzz**: Yazım hatalarına dayanıklı sorgu iyileştirme
- **Pandas**: İstatistiksel sorgular ve toplulaştırma

## 🎞️ Veri Seti

**TMDB 5000 Movies Dataset** - [Hugging Face](https://huggingface.co/datasets/AiresPucrs/tmdb-5000-movies)

- **Kaynak**: The Movie Database (TMDB)
- **Boyut**: ~5000 film
- **İçerik**: Film başlıkları, özetler, türler, oyuncular, ekip, yayın tarihleri, puanlar, popülerlik metrikleri
- **Dil**: İngilizce
- **Erişim**: Açık (Hugging Face Datasets üzerinden otomatik indirilir)
- **Kullanım**: Film RAG sistemi için belge kaynağı; aggregation sorguları için DataFrame

### Veri Seti Özellikleri
- ✅ Zengin metadata (genres, keywords, cast, crew, production companies)
- ✅ Oy sayısı ve puanlama bilgileri (güvenilir rating sorguları için)
- ✅ JSON formatında yapılandırılmış alanlar
- ✅ Otomatik indirme ve cache desteği

## 🚀 Kurulum

### 1. Gerekli Paketleri Yükleyin

```bash
# Virtual environment oluşturun (önerilir)
python -m venv genai-venv
genai-venv\Scripts\Activate.ps1  # Windows PowerShell
# source genai-venv/bin/activate  # macOS/Linux

# Paketleri yükleyin
pip install -r requirements.txt
```

### 2. API Anahtarlarını Ayarlayın

Proje kök dizininde `.env` dosyası oluşturun:

```env
GOOGLE_API_KEY=your_google_api_key_here
# Opsiyonel: veri seti indirme hızı için avantaj sağlar
HF_TOKEN=your_huggingface_token_here
```

- **Google API Key**: [Google AI Studio](https://makersuite.google.com/app/apikey) üzerinden alabilirsiniz
- **Hugging Face Token** (opsiyonel): [Hugging Face Settings](https://huggingface.co/settings/tokens) üzerinden alabilirsiniz

### 3. Uygulamayı Çalıştırın

```bash
streamlit run .\project.py
```

Tarayıcınızda otomatik olarak açılacaktır (genellikle http://localhost:8501).

## 📁 Proje Yapısı

```
.
├── project.py           # Ana uygulama dosyası (Türkçe yorumlarla)
├── requirements.txt     # Python bağımlılıkları
├── .env                 # API anahtarları (git'e eklenmez)
├── .gitignore           # Git ignore kuralları
├── README.md            # Bu dosya
└── venv/                # Sanal ortam (git'e dahil değil)
```

## 💡 Nasıl Çalışır?

1. **Veri Yükleme**: Hugging Face'ten TMDB 5000 Movies veri seti indirilir
2. **Belge Hazırlama**: Her film için zenginleştirilmiş içerik oluşturulur (başlık, özet, türler, oyuncular, ekip)
3. **Embedding**: Her belge İngilizce embedding modeli (all-mpnet-base-v2) ile vektöre dönüştürülür
4. **Vektör Veritabanı**: Vektörler InMemory document store'da saklanır
5. **Sorgu İyileştirme**: Kullanıcı sorusu RapidFuzz ile yazım hatalarına karşı güçlendirilir
6. **Sorgulama**: Sorgu embedding'e dönüştürülür ve en ilgili 20 belge bulunur
7. **Yanıt Üretimi**:   
   • **Aggregation sorguları** (highest-rated, most popular vb.) → Pandas ile istatistiksel yanıt  
   • **RAG sorguları** → Gemini modeli, bulunan belgelerden yararlanarak yanıt oluşturur

### 🔄 Pipeline Akışı

```
Kullanıcı Sorusu
    ↓
Sorgu İyileştirme (RapidFuzz)
    ↓
Aggregation mu? → EVET → Pandas İstatistiksel Yanıt
    ↓ HAYIR
Text Embedding (all-mpnet-base-v2)
    ↓
InMemoryEmbeddingRetriever (top_k=20)
    ↓
PromptBuilder (SMART RESPONSE FORMATTING)
    ↓
PromptToChatMessages
    ↓
Google Gemini (gemini-2.0-flash-exp)
    ↓
Final Answer
```

### 🔧 Ana Fonksiyonlar (`project.py`)

#### Veri ve Hazırlık
- **`load_and_prepare_data()`**: TMDB veri setini Hugging Face'ten indirir, her film için zenginleştirilmiş içerik (başlık varyasyonları, özet, türler, oyuncular, ekip) oluşturur ve Haystack Document listesi döndürür
- **`load_tmdb_dataframe()`**: Aynı veri setini Pandas DataFrame olarak yükler; aggregation sorguları için sayısal sütunları normalize eder (oy, puan, süre, yıl)

#### Vektör ve RAG Pipeline
- **`create_faiss_index(documents)`**: Belgeleri all-mpnet-base-v2 modeliyle vektörleştirir ve InMemoryDocumentStore'a yazar
- **`build_rag_pipeline(document_store)`**: Text embedder → retriever (top_k=20) → prompt builder → Gemini generator bileşenlerini birbirine bağlayan Haystack pipeline'ını oluşturur
- **`PromptToChatMessages`** (custom component): Düz metin prompt'u ChatMessage listesine çevirir (Gemini entegrasyonu için)

#### Sorgu İyileştirme ve Aggregation
- **`normalize_query(query, movie_titles)`**: RapidFuzz ile yazım hatalarına/farklı casing'e dayanıklı başlık eşleştirmesi yapıp sorguyu güçlendirir
- **`is_aggregation_query(query)`**: Sorgunun "highest-rated", "most popular" gibi bir sıralama sorgusu olup olmadığını tespit eder; metrik, sıralama yönü, tür, limit bilgilerini döndürür
- **`get_aggregated_results(df, params)`**: DataFrame üzerinde pandas ile sıralama/filtreleme yapar (rating için vote_count≥100 filtresi), biçimlendirilmiş sonuç döndürür

#### Yardımcı Fonksiyonlar
- **`get_movie_titles(documents)`**: Belgelerden film başlıklarını çıkarır (fuzzy matching için)

#### Streamlit Arayüzü
- **`main()`**: Tüm bileşenleri bir araya getirir; kullanıcı girdisini alır, aggregation/RAG yoluna yönlendirir, sidebar diagnostics gösterir, chat geçmişi yönetir

## 🎯 Örnek Sorular

### Seriler/Evrenler (Çoklu Film Listesi)
- "Star Wars movies"
- "Batman films"
- "Harry Potter series"

### Tek Film (Detaylı Bilgi)
- "Tell me about Inception"
- "What is The Dark Knight about?"
- "Show me details of Finding Nemo"

### Karşılaştırma/Sıralama
- "Highest-rated action movies"
- "Top 5 most popular sci-fi movies"
- "Longest movies"

### Sanatçı/Yönetmen
- "Movies by Christopher Nolan"
- "Films starring Leonardo DiCaprio"

## ⚠️ Önemli Notlar

- İlk çalıştırmada veri seti indirilir ve işlenir, bu işlem birkaç dakika sürebilir
- Embedding işlemi CPU'da yavaş olabilir; Streamlit cache mekanizması sayesinde sonraki çalıştırmalarda hızlı başlar
- Yanıtlar **İngilizce** üretilir (veri seti ve LLM optimizasyonu için)
- Kod içi yorumlar **Türkçe** eklenmiştir (bootcamp eğitim amaçlı)

## 📚 Kaynaklar

- [Haystack 2.x Dokümantasyonu](https://docs.haystack.deepset.ai/)
- [Google GenAI (Gemini) — Haystack Entegrasyonu](https://haystack.deepset.ai/integrations/google-genai)
- [Sentence-Transformers (all-mpnet-base-v2)](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
- [Hugging Face Datasets — TMDB 5000 Movies](https://huggingface.co/datasets/AiresPucrs/tmdb-5000-movies)
- [RapidFuzz](https://github.com/maxbachmann/RapidFuzz)

## 🔗 Uygulama Linki
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://movieassistantchatbot.streamlit.app/)

Canlı Demo: https://movieassistantchatbot.streamlit.app/
