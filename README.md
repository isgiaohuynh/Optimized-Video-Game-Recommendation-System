# 🎮 Video Game Recommendation System (Optimized Version)

## 📘 Project Overview
This project is the **final assignment** for the *Recommender Systems* course at the **University of Science – Vietnam National University, Ho Chi Minh City**.  
The requirement of the project was to **understand, reimplement, and optimize** an existing open-source system.  
Our team selected the following GitHub project as the base implementation:  
🔗 [Original Project – Video Game Recommendation Engine by SulmanK](https://github.com/SulmanK/Video-Game-Recommendation-Engine/blob/master/Video%20Game%20Recommendation%20Engine.ipynb)

Our optimized version focuses on improving **speed, flexibility, and diversity of recommendations** while maintaining interpretability and academic clarity.

---

## 🧠 Methodology

### 1. Dataset
The dataset was collected using the **Giant Bomb API**, containing information on over **37,000 video games** with 11 attributes:
- `id` – Unique identifier  
- `name` – Game title  
- `original_game_rating` – Age rating  
- `original_release_date` – Release date  
- `platform` – Compatible platforms  
- `developer` – Game developer  
- `genre` – Game genres  
- `theme` – Game themes  
- `concept` – Game concept or idea  
- `franchise` – Game series  
- `image_url` – Cover image  

Since the dataset lacks user interaction data (ratings, history, preferences), **Content-Based Filtering** is the most suitable approach.

---

### 2. Applied Techniques
- **TF-IDF Vectorization** – Encodes categorical text attributes (`genre`, `theme`, `developer`, `concept`, etc.) into numerical feature vectors.  
- **Cosine Similarity** – Measures similarity between games based on TF-IDF vectors.  
- **K-Nearest Neighbors (KNN)** – Finds the K most similar games for each input.  
- **Truncated SVD (Singular Value Decomposition)** – Reduces feature dimensionality to improve runtime performance.  
- **Attribute-Based Filtering** – Adds flexibility by recommending games that share selected attributes (`genre`, `theme`, or `concept`).
- **Web demo building**: Build an app demo for interactive recommendation visualization.

---

## ⚙️ Improvements over the Original Project
The original project provided a basic implementation of content-based recommendation using TF-IDF and cosine similarity.  
This optimized version introduces several major improvements:

| Optimization Aspect | Description |
|----------------------|-------------|
| **Runtime Efficiency** | Reduced average query time from **~13 seconds** to **0.05–0.2 seconds** using Truncated SVD. |
| **Modular Design** | Added parameterized functions (`tfidf_matrix`, `idf_list`, etc.) to precompute data and minimize redundant calculations. |
| **Filtering Extensions** | Integrated **conditional filtering** and **weighted scoring** for flexible and diverse recommendations. |
| **Data Handling** | Improved preprocessing for missing and noisy data. |
| **User Experience** | Enhanced relevance, speed, and diversity in recommendations. |

---

## 📊 Experimental Results

| Method | Avg. Runtime | Relevance | Diversity | Comment |
|--------|---------------|-----------|------------|----------|
| Cosine Similarity | ~12s | High | Moderate | Accurate but slow |
| KNN | ~7s | Fairly High | Good | Faster, more varied |
| **SVD + Cosine** | **0.18s** | High | Moderate | Balanced |
| **SVD + KNN** | **0.05s** | Fairly High | Excellent | ✅ Best performance |

> **Best-performing model:** SVD + KNN  
> Combines high diversity, instant response time, and stable similarity results.

---

## 🚀 Future Work
- Integrate **user-based or hybrid recommendation** models when user data becomes available.  
- Add **personalization** features where users can select or adjust preferred attributes.  
- Incorporate **popularity and rating scores** from external platforms to enhance ranking.  

---

## 👥 Contributor
- *Chiêm Huỳnh Giao*  
- *Nguyễn Ngọc Bảo Hân*  
- *Nguyễn Thị Xuân Hương*  

---

📅 *Completed on: June 29, 2025*  
🔗 *Dataset Source:* [Giant Bomb API](https://www.giantbomb.com/api/)  
🔗 *Original Project:* [Video Game Recommendation Engine by SulmanK](https://github.com/SulmanK/Video-Game-Recommendation-Engine/blob/master/Video%20Game%20Recommendation%20Engine.ipynb)
