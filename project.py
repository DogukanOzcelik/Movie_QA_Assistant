# -*- coding: utf-8 -*-

# =================================================================================
# TEK DOSYADA BİRLEŞTİRİLMİŞ PROJE KODU
# (TMDB 5000 Movies veri seti üzerinde RAG uygulaması - Uygulama çıktıları İngilizce)
#
# Bu dosya; veri yükleme, belge hazırlama, vektör veritabanı oluşturma,
# Haystack RAG pipeline kurulumu ve Streamlit arayüzünü tek bir yerde toplar.
# =================================================================================

import os
import json
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from datasets import load_dataset

# Haystack ve Gerekli Bileşenlerin İçe Aktarımı
from haystack import Pipeline, component
from haystack.dataclasses import Document, ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.writers import DocumentWriter
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import PromptBuilder
from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator
from haystack.utils import Secret
from rapidfuzz import fuzz, process

# --- Özel Bileşen: İstemi (prompt) ChatMessage'a çevir ---
@component
class PromptToChatMessages:
    """Verilen metin istemini (prompt) tek bir ChatMessage listesine çevirir.

    Amaç: Google GenAI jeneratörü mesaj listesi bekler; bu nedenle düz metin
    istemi ChatMessage nesnesine dönüştürülür.

    Girdi: prompt (str)
    Çıktı: messages (list[ChatMessage]) — tek kullanıcı mesajı içeren liste
    """

    @component.output_types(messages=list[ChatMessage])
    def run(self, prompt: str):
        """Kullanıcı mesajı olarak tek bir ChatMessage döndürür."""
        return {"messages": [ChatMessage.from_user(prompt)]}

# --- Yardımcı: Sorgu bir toplulaştırma/sıralama isteği mi? ---
def is_aggregation_query(query: str) -> dict:
    """Sorgunun sayısal bir kıyas/denge sıralaması gerektirip gerektirmediğini tespit eder.

    Dönen sözlük anahtarları:
    - is_agg: bool, gerçekten bir toplulaştırma/sıralama sorgusu mu?
    - metric: 'rating' | 'popularity' | 'runtime' | 'release_date'
    - order: 'asc' | 'desc'
    - genre: (varsa) tür filtresi, ör. "Action"
    - limit: kaç sonuç listelenecek (varsayılan 5)
    
    Girdi: query (str)
    Çıktı: dict (is_agg, metric, order, genre, limit)
    """
    query_lower = query.lower()
    
    result = {
        'is_agg': False,
        'metric': None,
        'order': 'desc',  # default: highest/most
        'genre': None,
        'limit': 5
    }
    
    # Toplulaştırma (aggregation) anahtar kelimelerini yakala
    agg_keywords = {
        'highest': ('rating', 'desc'),
        'best': ('rating', 'desc'),
        'top rated': ('rating', 'desc'),
        'highest rated': ('rating', 'desc'),
        'most popular': ('popularity', 'desc'),
        'popular': ('popularity', 'desc'),
        'longest': ('runtime', 'desc'),
        'shortest': ('runtime', 'asc'),
        'newest': ('release_date', 'desc'),
        'latest': ('release_date', 'desc'),
        'oldest': ('release_date', 'asc'),
        'lowest rated': ('rating', 'asc'),
        'worst': ('rating', 'asc'),
    }
    
    for keyword, (metric, order) in agg_keywords.items():
        if keyword in query_lower:
            result['is_agg'] = True
            result['metric'] = metric
            result['order'] = order
            break
    
    # Sorguda tür (genre) geçiyorsa yakala
    genres = ['action', 'comedy', 'drama', 'thriller', 'horror', 'romance', 
              'sci-fi', 'science fiction', 'fantasy', 'animation', 'documentary',
              'adventure', 'crime', 'mystery', 'war', 'western', 'musical']
    
    for genre in genres:
        if genre in query_lower:
            result['genre'] = genre.replace('sci-fi', 'science fiction').title()
            break
    
    # Kaç sonuç istendiğini (limit) tespit et
    import re
    numbers = re.findall(r'\b(\d+)\b', query_lower)
    if numbers:
        try:
            limit = int(numbers[0])
            if 1 <= limit <= 20:
                result['limit'] = limit
        except:
            pass
    
    # Yaygın desenler: "top 5", "first 3" vb.
    top_match = re.search(r'(?:top|first)\s+(\d+)', query_lower)
    if top_match:
        result['limit'] = int(top_match.group(1))
    
    return result

# --- Yardımcı: DataFrame üzerinden toplulaştırılmış sonuç üret ---
def get_aggregated_results(df: pd.DataFrame, agg_params: dict) -> str:
    """DataFrame üzerinde toplulaştırma/sıralama yapıp biçimlendirilmiş sonuç döndürür.

    Not: "rating" metriği için en az 100 oy şartı uygulanır; bu sayede tek tük
    oylarla şişmiş filmler liste başına gelmez.

    Girdi: df (pd.DataFrame), agg_params (dict)
    Çıktı: Sonuç metni (str) veya uygun değilse None
    """
    if df is None or not agg_params['is_agg']:
        return None
    
    metric = agg_params['metric']
    order = agg_params['order']
    genre = agg_params['genre']
    limit = agg_params['limit']
    
    # İstenen metriği ilgili sütun adına eşle
    metric_col_map = {
        'rating': 'vote_average',
        'popularity': 'popularity',
        'runtime': 'runtime',
        'release_date': 'year'
    }
    
    col = metric_col_map.get(metric)
    if col not in df.columns:
        return None
    
    # Tür filtresi istenmişse uygula (genres_norm: normalize edilmiş tür listesi)
    working_df = df.copy()
    if genre:
    # genres_norm bir liste sütunudur
        working_df = working_df[working_df['genres_norm'].apply(
            lambda g: any(genre.lower() in str(x).lower() for x in g)
        )]
    
    # İlgili metrik sütununda NaN olanları çıkar
    working_df = working_df[working_df[col].notna()]
    
    # ÖNEMLİ: Rating için minimum oy sayısı filtresi uygulayalım
    if metric == 'rating' and 'vote_count' in working_df.columns:
        working_df['vote_count'] = pd.to_numeric(working_df['vote_count'], errors='coerce')
    # Rating sorguları için en az 100 oy gerektir (daha güvenilir)
        working_df = working_df[working_df['vote_count'] >= 100]
    
    if len(working_df) == 0:
        return f"No movies found matching the criteria (with sufficient votes)."
    
    # Sıralama yönünü belirle ve sırala
    ascending = (order == 'asc')
    sorted_df = working_df.sort_values(by=col, ascending=ascending)
    
    # İlk N sonucu al
    top_movies = sorted_df.head(limit)
    
    # Sonuçları kullanıcıya okunabilir biçimde hazırla
    results = []
    metric_label = {
        'rating': 'Rating',
        'popularity': 'Popularity',
        'runtime': 'Runtime',
        'release_date': 'Year'
    }
    
    for idx, row in top_movies.iterrows():
        title = row.get('title', 'Unknown')
        value = row[col]
        
        if metric == 'runtime':
            value_str = f"{int(value)} min"
        elif metric == 'rating':
            value_str = f"{value:.1f}/10"
            # Show vote count for transparency
            vote_count = row.get('vote_count', 0)
            if vote_count:
                value_str += f" ({int(vote_count):,} votes)"
        elif metric == 'popularity':
            value_str = f"{value:.1f}"
        else:
            value_str = str(value)
        
    # Tür ve çıkış yılını dahil et
        year = row.get('year', 'N/A')
        results.append(f"- **{title}** ({year}) - {metric_label[metric]}: {value_str}")
    
    genre_text = f" in {genre}" if genre else ""
    order_text = "highest" if order == "desc" else "lowest"
    
    header = f"Here are the {order_text} {metric_label[metric].lower()} movies{genre_text}:\n\n"
    
    # Rating sorguları için not ekle (güvenilirlik)
    if metric == 'rating':
        header += "*Note: Showing movies with at least 100 votes for reliability.*\n\n"
    
    return header + "\n".join(results)

# --- Yardımcı: Sorguyu (başlık) bulanık eşleştirme ile normalize et ---
def normalize_query(query: str, movie_titles: list[str]) -> str:
    """Sorguyu (kullanıcı girişi) bilinen film başlıklarına göre bulanık eşleştirerek güçlendirir.

    Amaç: Yazım hataları/küçük-büyük harf farkı gibi durumlarda doğru filmi bulma
    şansını artırmak için, en yakın eşleşen başlıkları sorguya eklemek.

    Girdi: query (str), movie_titles (list[str])
    Çıktı: Geliştirilmiş/normalize edilmiş sorgu (str)
    """
    # Olası film başlık adaylarını çıkar (muhtemel başlık kelime grupları)
    words = query.split()
    potential_titles = []
    
    # Büyük harfle başlayan dizileri veya tırnak içi metinleri yakala
    current_title = []
    for word in words:
        clean_word = word.strip('",.:;!?()[]{}')
        if clean_word and (clean_word[0].isupper() or clean_word.lower() in ['the', 'a', 'an', 'of', 'and']):
            current_title.append(clean_word)
        else:
            if len(current_title) >= 1:  # En az 1 kelime
                potential_titles.append(' '.join(current_title))
            current_title = []
    if current_title:
        potential_titles.append(' '.join(current_title))
    
    # Tüm sorguyu da başlık adayı olarak dene
    potential_titles.append(query)
    
    # Bilinen başlık listesine karşı bulanık eşleştirme uygula
    best_matches = []
    for potential in potential_titles:
        if len(potential) >= 3:  # Minimum 3 karakter
            matches = process.extract(
                potential, 
                movie_titles, 
                scorer=fuzz.token_sort_ratio,
                limit=2,
                score_cutoff=75  # Daha iyi eşleşmeler için daha yüksek eşik
            )
            if matches:
                best_matches.extend([match[0] for match in matches])
    
    # Eşleşen başlıklarla sorguyu zenginleştir
    if best_matches:
        unique_matches = list(dict.fromkeys(best_matches[:2]))  # En iyi 2 eşleşme
        # Geri getirmeyi (retrieval) iyileştirmek için tam başlıkları ekle
        enhanced = f"{query}. Searching for movies: {', '.join(unique_matches)}"
        return enhanced
    
    return query



# --- Adım 1: Ortam Değişkenlerini ve API Anahtarını Yükleme ---
# .env dosyasını okuyarak Google API anahtarını güvenli şekilde alırız.
# Hugging Face Spaces'e dağıtırken bu anahtarı "Secrets" bölümüne ekleyin.
try:
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    HF_TOKEN = os.getenv("HF_TOKEN")
    if not GOOGLE_API_KEY:
        st.error("GOOGLE_API_KEY not found. Please check your .env file or Streamlit secrets settings.")
        st.stop()
except Exception as e:
    st.error(f"Error loading environment variables: {e}")
    st.stop()

# --- Adım 2: Veri Yükleme ve Hazırlama (TMDB 5000 Movies) ---
# Bu fonksiyon Hugging Face'ten TMDB 5000 Movies veri setini indirir, işler ve
# Haystack'in Document formatına çevirir. İçerik; film adı, özet, türler, anahtar
# kelimeler, oyuncu/ekip bilgileri ile zenginleştirilir. Streamlit cache mekanizması
# sayesinde ilk çalıştırmadan sonra tekrar indirilmez.
@st.cache_resource
def load_and_prepare_data():
    """TMDB 5000 Movies veri setini indirir ve Haystack Document listesi üretir."""
    with st.spinner("Loading TMDB 5000 Movies dataset..."):
        try:
            # 1) Veri setini yükle
            if HF_TOKEN:
                dataset = load_dataset("AiresPucrs/tmdb-5000-movies", split="train", token=HF_TOKEN)
            else:
                dataset = load_dataset("AiresPucrs/tmdb-5000-movies", split="train")
            df = dataset.to_pandas()

            # Yardımcı: Liste/JSON tipindeki alanlardan güvenli şekilde 'name' değerlerini çıkar
            def _ensure_list(value):
                if value is None or (isinstance(value, float) and pd.isna(value)):
                    return []
                if isinstance(value, list):
                    return value
                if isinstance(value, str):
                    v = value.strip()
                    # JSON dizesi olabilir
                    if (v.startswith("[") and v.endswith("]")) or (v.startswith("{") and v.endswith("}")):
                        try:
                            parsed = json.loads(v)
                            return parsed if isinstance(parsed, list) else [parsed]
                        except Exception:
                            return []
                    return []
                return []

            def _names_from(field_val):
                items = _ensure_list(field_val)
                names = []
                for it in items:
                    if isinstance(it, dict) and 'name' in it:
                        names.append(str(it['name']))
                return names

            def _cast_top_k(cast_val, k=5):
                items = _ensure_list(cast_val)
                result = []
                for it in items[:k]:
                    if isinstance(it, dict):
                        nm = str(it.get('name') or it.get('original_name') or "")
                        ch = str(it.get('character') or "")
                        if nm and ch:
                            result.append(f"{nm} ({ch})")
                        elif nm:
                            result.append(nm)
                return result

            def _crew_roles(crew_val, roles=("Director", "Screenplay", "Writer"), k=3):
                items = _ensure_list(crew_val)
                result = []
                for it in items:
                    if isinstance(it, dict):
                        job = str(it.get('job') or "")
                        if job in roles:
                            nm = str(it.get('name') or it.get('original_name') or "")
                            if nm:
                                result.append(f"{job}: {nm}")
                return result[:k]

            # Haystack Document nesnelerini oluştur
            documents = []
            for _, row in df.iterrows():
                title = str(row.get('title') or row.get('original_title') or "")
                overview = str(row.get('overview') or "")
                tagline = str(row.get('tagline') or "")
                release_date = str(row.get('release_date') or "")
                original_language = str(row.get('original_language') or "")
                original_title = str(row.get('original_title') or "")
                genres = _names_from(row.get('genres'))
                keywords = _names_from(row.get('keywords'))
                prod_companies = _names_from(row.get('production_companies'))
                prod_countries = _names_from(row.get('production_countries'))
                spoken_languages = _names_from(row.get('spoken_languages'))
                cast_list = _cast_top_k(row.get('cast'), k=5)
                crew_info = _crew_roles(row.get('crew'))
                runtime = row.get('runtime')
                vote_average = row.get('vote_average')
                popularity = row.get('popularity')

                # İçerik metni (LLM için tamamen İngilizce tutulur)
                parts = []
                header = []
                if title:
                    header.append(f"Title: {title}")
                if original_title and original_title != title:
                    header.append(f"Original Title: {original_title}")
                if header:
                    parts.append(" | ".join(header))
                
                # Aramada güçlendirme için başlığa farklı yazım varyasyonlarını da ekle
                if title:
                    parts.append(f"Movie Title: {title}")
                    parts.append(f"TITLE: {title.upper()}")  # Uppercase version
                    parts.append(f"title: {title.lower()}")  # Lowercase version
                if original_title and original_title != title:
                    parts.append(f"Original Title: {original_title}")
                    parts.append(f"Alternative Title: {original_title}")
                    
                if overview:
                    parts.append(f"Overview: {overview}")
                if tagline:
                    parts.append(f"Tagline: {tagline}")
                if genres:
                    parts.append(f"Genres: {', '.join(genres)}")
                if keywords:
                    parts.append(f"Keywords: {', '.join(keywords[:15])}")
                if cast_list:
                    parts.append(f"Cast: {', '.join(cast_list)}")
                if crew_info:
                    parts.append(f"Crew: {', '.join(crew_info)}")
                if prod_companies:
                    parts.append(f"Production Companies: {', '.join(prod_companies[:10])}")
                if prod_countries:
                    parts.append(f"Countries: {', '.join(prod_countries)}")
                if spoken_languages:
                    parts.append(f"Languages: {', '.join(spoken_languages)}")
                details = []
                if release_date and release_date.lower() != 'nan':
                    details.append(f"Release Date: {release_date}")
                if original_language and original_language.lower() != 'nan':
                    details.append(f"Language: {original_language}")
                if runtime and not (isinstance(runtime, float) and pd.isna(runtime)):
                    details.append(f"Runtime: {int(runtime)} min")
                if vote_average and not (isinstance(vote_average, float) and pd.isna(vote_average)):
                    details.append(f"Rating: {vote_average}")
                if popularity and not (isinstance(popularity, float) and pd.isna(popularity)):
                    details.append(f"Popularity: {popularity}")
                if details:
                    parts.append("Details: " + "; ".join(details))

                content = "\n".join(parts).strip()

                meta = {
                    'title': title if title else "Unknown Movie",
                    'release_date': release_date,
                    'original_language': original_language
                }

                if content:
                    documents.append(Document(content=content, meta=meta))

            # Film kayıtları kısa ve yapısal; parçalamaya (split) gerek yok
            return documents
        except Exception as e:
            st.error(f"TMDB veri seti yüklenirken/işlenirken hata: {e}")
            return None

# Ham DataFrame yükleyici: İstatistik/toplulaştırma (aggregation) soruları için kullanılır
@st.cache_resource
def load_tmdb_dataframe():
    """TMDB 5000 Movies veri setini pandas DataFrame olarak yükler ve sütunları normalize eder.

    Çıktı: Pandas DataFrame (türler normalize edilmiş, yıl sütunu eklenmiş)
    """
    try:
        if HF_TOKEN:
            dataset = load_dataset("AiresPucrs/tmdb-5000-movies", split="train", token=HF_TOKEN)
        else:
            dataset = load_dataset("AiresPucrs/tmdb-5000-movies", split="train")
        df = dataset.to_pandas()
    # Tip dönüşümleri (sayısal alanlar)
        for col in ["vote_average", "popularity", "runtime", "vote_count"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        if 'release_date' in df.columns:
            df['release_date_parsed'] = pd.to_datetime(df['release_date'], errors='coerce')
            df['year'] = df['release_date_parsed'].dt.year

    # Yardımcı: JSON string/list'ten name alanlarını çıkar ve normalize et
        def _ensure_list(value):
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return []
            if isinstance(value, list):
                return value
            if isinstance(value, str):
                v = value.strip()
                if (v.startswith("[") and v.endswith("]")) or (v.startswith("{") and v.endswith("}")):
                    try:
                        parsed = json.loads(v)
                        return parsed if isinstance(parsed, list) else [parsed]
                    except Exception:
                        return []
                return []
            return []

        def _names_from(v):
            items = _ensure_list(v)
            names = []
            for it in items:
                if isinstance(it, dict) and 'name' in it:
                    names.append(str(it['name']))
            return names

        def _norm_tr(s: str) -> str:
            s = s.lower()
            repl = {
                'ç': 'c', 'ğ': 'g', 'ı': 'i', 'i̇': 'i', 'ö': 'o', 'ş': 's', 'ü': 'u',
                'â': 'a', 'ê': 'e', 'î': 'i', 'ô': 'o', 'û': 'u'
            }
            for k, v in repl.items():
                s = s.replace(k, v)
            return s

        if 'genres' in df.columns:
            df['genres_norm'] = df['genres'].apply(lambda v: [_norm_tr(n) for n in _names_from(v)])
        else:
            df['genres_norm'] = [[] for _ in range(len(df))]
        return df
    except Exception as e:
        st.warning(f"Ham tablo yüklenirken hata: {e}")
        return None

# --- Bulanık eşleşme için tüm film başlıklarını çıkar ---
@st.cache_resource
def get_movie_titles(_documents):
    """Bulanık eşleştirmede kullanılmak üzere tüm film başlıklarını döndürür.

    Girdi: _documents (list[Document])
    Çıktı: Başlık listesi (list[str])
    """
    if not _documents:
        return []
    titles = []
    for doc in _documents:
        if hasattr(doc, 'meta') and 'title' in doc.meta:
            titles.append(doc.meta['title'])
    return titles

# --- Adım 3: Vektör Veritabanı Oluşturma ---
# Hazırlanan belgeler, SentenceTransformers gömme modeliyle vektörleştirilir ve
# InMemoryDocumentStore üzerinde saklanır (FAISS benzeri amaca hizmet eder).
# Bu işlem cache'lenir; tekrar tekrar çalıştırılmaz.
@st.cache_resource
def create_faiss_index(_split_docs):
    """Verilen belgeler için bellek içi bir DocumentStore oluşturup gömmeleri yazar.

    Girdi: _split_docs (list[Document])
    Çıktı: InMemoryDocumentStore veya None
    """
    if not _split_docs:
        return None
        
    with st.spinner("Creating vector database and processing documents..."):
        try:
            document_store = InMemoryDocumentStore()
            
            # Doküman tarafı için güçlü bir gömme modeli
            doc_embedder = SentenceTransformersDocumentEmbedder(
                model="sentence-transformers/all-mpnet-base-v2"
            )

            # Belgeleri ve gömmeleri saklamaya yazacak küçük bir indeksleme hattı (pipeline)
            indexing_pipeline = Pipeline()
            indexing_pipeline.add_component("embedder", doc_embedder)
            indexing_pipeline.add_component("writer", DocumentWriter(document_store=document_store))
            indexing_pipeline.connect("embedder.documents", "writer.documents")

            # İndeksi oluşturmak için hattı çalıştır
            indexing_pipeline.run({"embedder": {"documents": _split_docs}})
            
            return document_store
        except Exception as e:
            st.error(f"Vektör indeksi oluşturulurken hata: {e}")
            return None

# --- Adım 4: RAG Pipeline Kurulumu ---
# Retriever + PromptBuilder + PromptToChatMessages + GoogleGenAIChatGenerator bileşenleri
# bir araya getirilerek sorgulanabilir bir Haystack hattı (pipeline) oluşturulur.
@st.cache_resource
def build_rag_pipeline(_document_store):
    """Verilen document_store üzerinde tam bir RAG hattı (pipeline) kurar.

    Girdi: _document_store (InMemoryDocumentStore)
    Çıktı: Pipeline veya None
    """
    if not _document_store:
        return None
        
    try:
    # 1. Retriever: Serilerde/evrenlerde birden fazla filmi yakalamak için top_k=20
        retriever = InMemoryEmbeddingRetriever(document_store=_document_store, top_k=20)

    # 2. İstem (Prompt) Şablonu (tamamen İngilizce - Gemini için optimize)
        template = """You are a helpful movie assistant. Answer the question based ONLY on the provided movie information below.

IMPORTANT RULES:
- Use ONLY the information from the provided documents below
- If information is not available, say "I don't have enough information about this"
- Answer in English only

SMART RESPONSE FORMATTING:
- If MULTIPLE related movies are found (e.g., Star Wars films, Batman movies):
  * List ALL of them in chronological order
  * For each: Title, Year, Rating, Brief description/genres
  * Format as a numbered list
  
- If asking about ONE specific movie (e.g., "What is Inception about?"):
  * Provide COMPREHENSIVE details:
    - Title and original title
    - Overview/plot summary
    - Genres, Release date, Runtime
    - Rating and popularity
    - Cast and Crew
    - Production details
  * Use organized sections with headers

- For comparison queries (highest rated, most popular, year):
  * List movies with relevant metrics
  * Include title, year, rating

{% for doc in documents %}
--- Movie {{ loop.index }} ---
Title: {{ doc.meta['title'] }}
{{ doc.content }}

{% endfor %}

Question: {{question}}
Answer:"""
    # PromptBuilder için gereken değişkenleri belirt
        prompt_builder = PromptBuilder(template=template, required_variables=["documents", "question"])

        # 3. Üreteç (Generator) — Daha tutarlı/ayrıntılı cevaplar için üretim parametreleri
        generation_kwargs = {
            "temperature": 0.3,  # Daha tutarlı ve doğru cevaplar için daha düşük sıcaklık
            "top_p": 0.85,      # Daha odaklı token seçimi için
            "max_output_tokens": 1200,  # Ayrıntılı film bilgileri için artırıldı
        }
        generator = GoogleGenAIChatGenerator(
            model="gemini-2.0-flash",  # Güncel model
            api_key=Secret.from_env_var("GOOGLE_API_KEY"),
            generation_kwargs=generation_kwargs
        )

        # Sorgular için metin gömme modeli (doküman embedder ile aynı olmalı)
        text_embedder = SentenceTransformersTextEmbedder(
            model="sentence-transformers/all-mpnet-base-v2"
        )

        # İstemi ChatMessage listesine çeviren bileşen
        prompt_to_messages = PromptToChatMessages()

        # 4. RAG hattını (pipeline) oluştur
        rag_pipeline = Pipeline()
        rag_pipeline.add_component("text_embedder", text_embedder)
        rag_pipeline.add_component("retriever", retriever)
        rag_pipeline.add_component("prompt_builder", prompt_builder)
        rag_pipeline.add_component("prompt_to_messages", prompt_to_messages)
        rag_pipeline.add_component("generator", generator)

        # Bileşenleri birbirine bağla (veri akışı)
        rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
        rag_pipeline.connect("prompt_builder.prompt", "prompt_to_messages.prompt")
        rag_pipeline.connect("prompt_to_messages.messages", "generator.messages")

        return rag_pipeline
    except Exception as e:
        st.error(f"RAG hattı oluşturulurken hata: {e}")
        return None

# --- Adım 5: Streamlit Web Arayüzü ---
def main():
    # Sayfa başlığı ve ikon
    st.set_page_config(
        page_title="TMDB Movie Q&A Assistant",
        page_icon="🎬",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    st.title("🎬 TMDB Movie Q&A Assistant")
    st.markdown(
        """
        <div style='background-color: #f0f2f6; padding: 0.6rem 1rem; border-radius: 0.5rem; margin-bottom: 0.75rem;'>
        <p style='margin: 0; color: #31333F;'>
        📚 <strong>Akbank Generative AI Bootcamp</strong> — RAG-based movie assistant<br>
        🎯 Dataset: <code>AiresPucrs/tmdb-5000-movies</code> | 🤖 Generator: Google Gemini
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Örnek sorular (iki kolon)
    with st.expander("💡 Example Questions"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **🎭 Series/Franchise:**
            - Star Wars movies
            - Batman films
            - Harry Potter series
            
            **🎬 Specific Movie:**
            - Tell me about Inception
            - What is The Dark Knight about?
            """)
        with col2:
            st.markdown("""
            **📊 Comparison & Rankings:**
            - Highest-rated action movies
            - Top 5 most popular sci-fi movies
            
            **🎥 Director & Cast:**
            - Movies by Christopher Nolan
            """)

    # Gerekli bileşenleri yükle/cache'le
    split_documents = load_and_prepare_data()
    # (Sidebar kullanımı kaldırıldı — daha kompakt ve ortalanmış görünüm için)
    
    # Bulanık eşleşme için film başlıklarını al
    movie_titles = get_movie_titles(split_documents) if split_documents else []
    
    # Toplulaştırma sorguları için DataFrame'i yükle
    tmdb_df = load_tmdb_dataframe()
    
    if split_documents:
        document_store = create_faiss_index(split_documents)
        if document_store:
            rag_pipeline = build_rag_pipeline(document_store)
        else:
            rag_pipeline = None
    else:
        rag_pipeline = None

    if not rag_pipeline:
        st.warning("Application failed to start. Please check the error messages.")
        st.stop()

    # Sohbet geçmişini oturum durumunda sakla
    if "messages" not in st.session_state:
        st.session_state.messages = list()

    # Sohbet kontrol çubuğu: Temizle ve mesaj sayacı
    col_clear, col_count = st.columns([1, 1])
    with col_clear:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col_count:
        st.markdown(f"<div style='text-align:right; font-size:1.1em;'>💬 <b>{len(st.session_state.messages) // 2}</b> messages</div>", unsafe_allow_html=True)

    # Önceki mesajları göster
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Kullanıcıdan girdi al
    if prompt := st.chat_input("Ask about any movie... (e.g., What is Inception about?)"):
        # Kullanıcının mesajını geçmişe ekle ve göster
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Bu bir toplulaştırma/sıralama sorgusu mu?
        agg_params = is_aggregation_query(prompt)
        
        if agg_params['is_agg'] and tmdb_df is not None:
            # DataFrame ile toplulaştırma sorgusunu ele al
            with st.spinner("📊 Analyzing movie data..."):
                response = get_aggregated_results(tmdb_df, agg_params)
                
                if not response:
                    response = "I couldn't perform the requested analysis. Please try rephrasing your question."
                
                # Kenar çubuğunda (sidebar) tanılama bilgisi göster
                # Ayrıntılar gizlendi (sidebar kaldırıldı)
        
        else:
            # Normal RAG sorgusunu işle
            # Daha iyi eşleşme için sorguyu normalize et (bulanık başlık eşleştirme)
            normalized_prompt = normalize_query(prompt, movie_titles) if movie_titles else prompt
            
            # Kenar çubuğunda iyileştirilmiş sorguyu göster
            # İsteğe bağlı: Sorgu iyileştirme mesajlarını göstermiyoruz (daha kısa görünüm)

            # RAG hattını çalıştır ve yanıtı al
            with st.spinner("🔎 Searching movie information and generating answer..."):
                try:
                    result = rag_pipeline.run({
                        "text_embedder": {"text": normalized_prompt},
                        "prompt_builder": {"question": prompt}  # Use original for the question
                    })
                    
                    response = "An error occurred or no response was received."
                    
                    if result and "generator" in result and result["generator"]["replies"]:
                        # ChatMessage nesnesinden metin içeriğini çıkar
                        chat_message = result["generator"]["replies"][0]
                        
                        # _content: TextContent nesnelerini içeren bir listedir
                        if hasattr(chat_message, '_content') and chat_message._content:
                            response = chat_message._content[0].text.strip()
                        elif hasattr(chat_message, 'content'):
                            response = chat_message.content.strip()
                        else:
                            response = str(chat_message).strip()
                        
                        # Cevap boş ya da çok kısa ise kullanıcıya yardımcı mesaj ver
                        if not response or len(response) < 10:
                            response = "I'm sorry, I couldn't find enough information about your question. Please try a more specific question."
                            
                    # Tanılama bilgisi (opsiyonel - kenar çubuğunda gösterilir)
                    # Alınan belgeler paneli kaldırıldı (daha kısa görünüm)

                except Exception as e:
                    response = f"❌ An error occurred while processing your query: {str(e)}\n\nPlease try rephrasing your question."
                    st.error(f"Technical details: {e}")

        # Asistanın yanıtını sohbet geçmişine ekle ve göster
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

if __name__ == "__main__":
    main()