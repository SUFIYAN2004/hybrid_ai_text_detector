# 🔍 DetectAI: Hybrid Text Authenticity Detection System

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

DetectAI is a dual-model system designed to verify if a text is **human-written** or **AI-generated**. By combining traditional Machine Learning with Deep Learning, the system provides a robust and reliable analysis of digital content.

## 🚀 Live Demo
[Link to your Render URL here]

---

## 💡 Key Features
- **Dual-Model Analysis:** Get independent results from a Machine Learning (TF-IDF + Classifier) and a Deep Learning (Neural Network) model for comparison.
- **High Accuracy:** Achieved a peak performance of **99.7% accuracy** during testing.
- **Fast Performance:** Analysis completed in approximately **0.4 seconds**.
- **Modern UI:** Responsive dashboard built with Tailwind CSS, Bootstrap, and JavaScript.

## 🛠️ Tech Stack
- **Backend:** Python, Flask
- **AI/ML:** TensorFlow/Keras, Scikit-learn, NLTK, NumPy, Pandas
- **Frontend:** HTML5, CSS3, JavaScript, Tailwind CSS, Bootstrap
- **Dataset:** 20,000-row Kaggle AI Text Detection dataset

---

## 🏗️ Architecture & How It Works
1. **Input:** User pastes text into the web interface.
2. **ML Path:** Processes text using **TF-IDF vectorization** and a trained classifier.
3. **DL Path:** Processes text using **tokenization** and a **Neural Network**.
4. **Output:** Displays separate confidence scores and predictions for both models.

## 💻 Getting Started

### Prerequisites
- Python 3.9+
- 4GB RAM (recommended)

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/SUFIYAN2004/hybrid_ai_text_detector.git](https://github.com/SUFIYAN2004/hybrid_ai_text_detector.git)
   cd hybrid_ai_text_detector
  
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py

```
Access the app at http://127.0.0.1:5000

🔮 Future Enhancements
[ ] Implement an Ensemble Score to combine ML and DL results.
[ ] Add Document Upload (.pdf, .docx) support.
[ ] Multilingual detection capabilities.
[ ] User authentication and history tracking.

👤 Author
Mohammed Sufiyan
LinkedIn: https://www.linkedin.com/in/mohammedsufiyanv/

⭐ If you find this project helpful, please give it a star!


### 🎨 Finishing Touches:
1.  **Add a Screenshot:** I highly recommend taking the image from **Page 5** of your PDF and saving it as `screenshot.png` in your repo. Then, add `![DetectAI Screenshot](screenshot.png)` at the top of the README.
2.  **Add your LinkedIn Link:** This connects your GitHub directly to the LinkedIn article we wrote earlier.
**Would you like me to help you create a professional "About Me" section for your GitHub profile as
