# 🌿 Leaf Disease Detection System – Raitha Samparka App

An end-to-end **Leaf Disease Detection Application** designed to help farmers identify plant diseases early using **Computer Vision** and **Deep Learning**.

---

## 📸 Overview

This project provides a complete workflow:
- Capture or upload leaf images  
- Preprocess images (denoising, resizing, normalization)  
- Segment diseased regions using OpenCV  
- Predict disease class using a trained CNN model  
- Calculate disease severity percentage  
- Display results through an intuitive Tkinter GUI  

---

## 🚀 Features

- 📷 **Capture Image** using webcam  
- 🖼️ **Read/Upload Image**  
- 🧹 **Preprocessing** (noise removal, normalization, resizing)  
- ✂️ **Segmentation** using OpenCV  
- 🧠 **CNN-based Prediction**  
- 📊 **Disease Percentage Calculation**  
- 🗂️ **SQLite Database Integration**  
- 🖥️ **Tkinter GUI** for easy user interaction  

---

## 🛠 Tech Stack

- **Python 3**  
- **OpenCV**  
- **TensorFlow / Keras**  
- **Tkinter**  
- **Pillow (PIL)**  
- **NumPy**  
- **Scikit-image**  
- **SQLite**  

---

## 📁 Project Structure

```
Leafdisease/
│ main.py  
│ readimg.py  
│ preprocessing.py  
│ seg.py  
│ cnn.py  
│ perdet.py  
│ captimg.py  
│ Form.db  
│ back.png  
│ models/  
│ └── leaf_disease_classifier.keras  
│ train/  
│ test/  
```

---

## 🧠 How It Works

1. **User selects or captures an image**  
2. Image goes through **preprocessing pipeline**  
3. **Segmentation** identifies diseased areas  
4. Trained **CNN model** predicts leaf disease  
5. Percentage of disease spread is calculated  
6. GUI displays output + segmented image  

---

## ▶️ Running the Project

Install dependencies:

```
pip install -r requirements.txt
```

Run main app:

```
python main.py
```

---

## 📌 Next Improvements

- Integrate transfer learning (MobileNet, EfficientNet)  
- Add more crop datasets  
- Build a web version (Flask/React)  
- Convert desktop app into mobile app  

---

## 💡 Motivation

Agriculture is deeply affected by plant diseases, especially for small farmers.  
This tool aims to provide a **fast, offline, and accurate system** for disease detection to help farmers diagnose problems earlier and improve crop yield.

---

## 🤝 Contributions

Feel free to fork this repo, open issues, or submit PRs!

---

## 📬 Contact

**Amit Birbitte**  
📧 Email: —  amitbirbitte99@gmail.com
🔗 LinkedIn: —  https://www.linkedin.com/in/amit-birbitte-499657260/

---

## ⭐ Show Your Support

If you like this project, consider giving it a ⭐ on GitHub!
