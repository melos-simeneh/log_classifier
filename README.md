# 🚀 Log Classification API

A FastAPI-based web service for **automatic log message classification**. The API uses a trained Logistic Regression model with TF-IDF features to classify logs into actionable categories such as `SecurityIncident`, `ApplicationError`, and more.

## 📁 Project Structure

```bash
log_classifier/
├── main.py                  # FastAPI app to serve the classifier
├── model/
│   ├── train_model.py       # Script to train and save the model
│   └── classifier.pkl       # Trained model and TF-IDF vectorizer
├── schemas.py               # Pydantic schema for request body
└── requirements.txt         # Python dependencies
```

## 📚 Log Categories

The model classifies logs into the following categories:

- SystemEvent
- SecurityIncident
- Warning
- ApplicationError
- Debugging
- SystemFailure
- Informational

## 🚀 Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/melos-simeneh/log_classifier.git
cd log_classifier
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model (if needed)

```bash
python model/train_model.py
```

## 🧪 Run the API Server

```bash
uvicorn main:app --reload
```

Open your browser at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for Swagger UI.

## 📮 API Usage

**POST** `/classify`

**Request Example:**

```json
{
  "message": "SSL certificate expired"
}
```

**Response Example:**

```json
{
  "log_type": "SecurityIncident"
}
```

## 🧰 Dependencies

Ensure the following are listed in requirements.txt:

```nginx
fastapi
uvicorn
scikit-learn
joblib
Install with:
```

```bash
pip install -r requirements.txt
```

## 🧠 Notes

The classifier uses a small static dataset. For production, consider training with real-world log data.

The TF-IDF + Logistic Regression pipeline is saved to model/classifier.pkl

## 📬 Contact

Made with 💚 by **MELOS**

If you have any questions, feedback, feel free to reach out: [melos.simeneh@gmail.com](mailto:melos.simeneh@gmail.com)
