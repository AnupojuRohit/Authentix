# 🚀 Authentix AI — Fake Brand Detection System

Authentix is an AI-powered platform that detects whether a product is **authentic or counterfeit** using advanced computer vision and deep learning techniques.

---

## 🧠 Core Idea

Upload a product image → AI analyzes it → Get an **authenticity score + brand verification**

---

## ⚙️ Tech Stack

### 🔹 Frontend

* Next.js
* TypeScript
* Tailwind CSS

### 🔹 Backend

* FastAPI (Python)
* REST API architecture

### 🔹 AI / ML

* YOLOv8 → Logo detection
* CLIP → Image embeddings
* FAISS → Similarity search
* OpenCV → Image processing

---

## 🧩 Features

* 📸 Upload product images
* 🏷️ Brand selection
* 🧠 AI-powered authenticity scoring
* 🔍 Logo + pattern + typography analysis
* ⚡ Fast inference (< 2 seconds)
* 🛡️ Real-time verification system

---

## 📁 Project Structure

```
Authentix/
│
├── backend/
│   ├── app/
│   │   ├── routes/
│   │   ├── services/
│   │   ├── models/
│   │   └── main.py
│   ├── ml/
│   │   ├── preprocess.py
│   │   ├── train_logo_detector.py
│   │   ├── train_embedding_index.py
│   │   └── auto_cluster_brands.py
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── services/
│   │   └── app/
│   └── package.json
│
├── dataset/ (ignored in Git)
├── .gitignore
└── README.md
```

---

## 🚀 Setup Instructions

### 1️⃣ Clone Repository

```bash
git clone https://github.com/AnupojuRohit/Authentix.git
cd Authentix
```

---

### 2️⃣ Backend Setup

```bash
cd backend
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Run backend:

```bash
uvicorn app.main:app --reload
```

Backend runs at:
👉 http://127.0.0.1:8000

---

### 3️⃣ Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at:
👉 http://localhost:3000

---

## 🔗 API Endpoints

| Endpoint   | Method | Description                            |
| ---------- | ------ | -------------------------------------- |
| `/health`  | GET    | Check API status                       |
| `/predict` | POST   | Upload image & get authenticity result |

---

## 🧪 ML Pipeline

1. Dataset preprocessing
2. Brand clustering (unsupervised)
3. CLIP embedding generation
4. FAISS index creation
5. YOLOv8 logo detection
6. Similarity scoring

---

## ⚡ Current Status

* ✅ UI Completed
* ✅ Backend API Ready
* ✅ Dataset Processing Pipeline
* 🔄 ML Model Integration (In Progress)

---

## 🔮 Future Improvements

* Improve model accuracy
* Add more brand datasets
* Deploy on cloud (AWS/GCP)
* Real-time mobile integration
* Blockchain verification (optional)

---

## 👨‍💻 Author

**Rohit Anupoju**
B.Tech CSE | AI & Automation Enthusiast

---

## ⭐ Contribute

Feel free to fork this repo and contribute!

---

## 📌 Note

Dataset and trained models are not included in this repository due to size limitations.
