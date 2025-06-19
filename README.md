# Setup Guide

> 📌 **Note:** Before proceeding, make sure you have collected and populated the data into MongoDB **as specified in** [HPE-Data-Generator](https://github.com/VarshithPawarHR/HPE-Data-Generator/tree/main). This is required for the system to work correctly.

## I. Backend

### Requirements

- Python **3.10 - 3.12**
- MongoDB (local or cloud)
- `pip` for installing Python packages

### 1. Install Python
Download and install Python from the official site:  
🔗 [https://www.python.org/downloads/](https://www.python.org/downloads/)
- Use **Python 3.10** or **3.11**, or **3.12**. TensorFlow does not yet support Python 3.13.
### 2. Clone the Repository

```bash
git clone https://github.com/VarshithPawarHR/HPE-StoragePrediction

```
### 3. Open Project in VS Code

- Launch **Visual Studio Code**
- Open the folder you just cloned



### 4. Select Python Interpreter (VS Code GUI)

- Press <kbd>Ctrl</kbd> + <kbd>Shift</kbd> + <kbd>P</kbd>
- Select: `Python: Select Interpreter`
- Choose a Python version between **3.10 – 3.12**
- Avoid selecting **Python 3.13**



### 5. Create and Activate Virtual Environment

#### Option 1: **Using VS Code GUI**

- Press <kbd>Ctrl</kbd> + <kbd>Shift</kbd> + <kbd>P</kbd>
- Select: `Python: Create Environment`
- Choose:
  - Environment Type: `venv`
  - Python Interpreter: `3.10`, `3.11`, or `3.12`

> VS Code will create and activate the environment automatically.

#### Option 2: **Using Terminal**

```bash
# On Windows
py -3.12 -m venv venv

# On Linux/macOS
python3.12 -m venv venv

# Activate the virtual environment
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```
### 6. Install Dependencies

```bash
cd backend
pip install -r requirenments.txt
```

### 7. Set Environment Variables

Create a `.env` file inside the `backend` folder: (refer `.env.example`)

```bash
TF_ENABLE_ONEDNN_OPTS=0
MONGO_URL=mongodb+srv://<username>:<password>@<cluster-url>/<database>
MONGO_DB=your_db_name
MONGO_COLLECTION=your_collection
```

### 8. Run the Backend

```bash
fastapi dev main.py
```

FastAPI will be live at: `http://127.0.0.1:8000/`


### 9. Test API and Train Models (Swagger UI)

To access the API interface, open your browser and go to:  
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)



#### Trigger Training

Use the following endpoints to trigger model training:

| Endpoint               | Description          |
|------------------------|----------------------|
| `POST /retrain/daily`     | Train Daily Model     |
| `POST /retrain/weekly`    | Train Weekly Model    |
| `POST /retrain/monthly`   | Train Monthly Model   |
| `POST /retrain/quarterly` | Train Quarterly Model |

>  After hitting any of these endpoints, training will begin using the data in your MongoDB collection.

---

#### Model & Scaler Output

Once training is complete:

- The **trained model files** will be saved in the `models/` directory.
- The **scalers** used for preprocessing will be stored in the `scalers/` directory.

---

### Next Step: Setup Frontend

After your models are trained and saved, proceed to set up the **frontend dashboard** to visualize predictions and analytics.


---

