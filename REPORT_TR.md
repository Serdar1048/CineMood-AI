# Proje Raporu: IMDB Duygu Analizi (Sentiment Analysis)

## 1. Proje Özeti
Bu proje, Kaggle üzerinden temin edilen [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) verisetini kullanarak metin verilerini sınıflandırmayı amaçlar. Kullanıcı yorumlarının duygu durumunu (pozitif/negatif) tahmin etmek için üç farklı Ardışık Sinir Ağı (RNN) mimarisi kıyaslanmıştır.

## 2. Metodoloji ve İş Akışı

### A. Veri Ön İşleme (Data Preprocessing)
- **HTML Temizliği:** BeautifulSoup ile gereksiz etiketler kaldırıldı.
- **Normalizasyon:** Küçük harf dönüşümü ve noktalama işaretlerinin temizliği yapıldı.
- **Stopwords:** Anlamsız kelimeler elendi.
- **Lemmatization:** Kelimeler köklerine indirgenerek veri karmaşıklığı azaltıldı.

### B. Vektörizasyon (Vectorization)
- **Tokenizer:** En sık geçen 10.000 kelimelik bir sözlük oluşturuldu.
- **Padding:** Her yorum 500 kelime uzunluğuna sabitlendi.

## 3. Modeller ve Mimari Karşılaştırması

Bu projede üç temel mimari eğitilmiş ve performansları kıyaslanmıştır:

### 1. Simple RNN (Baseline)
- **Mantık:** Veriyi tek bir yönde, kelime kelime işler.
- **Sorun:** "Vanishing Gradient" (Gradyan Kaybolması) sorunu nedeniyle uzun cümlelerin başındaki bilgileri sonuna taşıyamaz.
- **Sonuç:** Kısa cümlelerde hızlıdır ancak uzun ve detaylı yorumlarda başarısı oldukça düşüktür. Projemizde temel bir kıyaslama (benchmark) noktasıdır.

### 2. GRU (Gated Recurrent Unit)
- **Mantık:** LSTM'in daha modern ve basitleştirilmiş halidir. Sadece iki kapı (Update ve Reset Gate) kullanır.
- **Neden Kullanıldı?:** LSTM'den daha az parametreye sahip olduğu için daha hızlı eğitilir.
- **Sonuç:** Performansı çoğu zaman LSTM'e çok yakındır ancak işlem gücü olarak daha verimlidir.

### 3. LSTM (Long Short-Term Memory) - Projenin Ana Motoru
- **Mantık:** Üç farklı kapı (Forget, Input, Output) kullanarak bilginin ne kadar süreyle "hafızada" tutulacağını kontrol eder.
- **Neden Kullanıldı?:** Film yorumlarındaki "çok uzun mesafeli bağımlılıkları" (cümlenin başındaki bir olumsuzluk ekinin sonu etkilemesi gibi) en iyi yakalayan modeldir.
- **Sonuç:** En tutarlı ve yüksek doğruluk oranını genellikle LSTM mimarisi sağlamıştır.

## 4. Sonuç Analizi ve Karşılaştırma

| Model | Hız | Uzun Mesafeli Hafıza | Tahmin Başarısı |
| :--- | :--- | :--- | :--- |
| **Simple RNN** | ⭐⭐⭐ | ⭐ | Düşük |
| **GRU** | ⭐⭐ | ⭐⭐⭐ | Yüksek |
| **LSTM** | ⭐ | ⭐⭐⭐⭐ | Çok Yüksek |

**Neden LSTM ve GRU daha iyi sonuç verdi?**
Film yorumları genellikle "Initially I thought it was bad, but then..." gibi karmaşık ve bir kelimenin tüm anlamı değiştirdiği yapılar içerir. Simple RNN bu tür "ama" (but) sonrası geçişlerini ve uzun süreli bağlamı takip edemezken, LSTM ve GRU bu bilgiyi kapı mekanizmaları sayesinde hafızasında (cell state) tutarak doğru duyguya ulaşabilmektedir.
