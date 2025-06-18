# Setup Guide
> 📌 **Note:** Before proceeding, make sure you have collected and populated the data into MongoDB **as specified in** [HPE-Data-Generator](https://github.com/VarshithPawarHR/HPE-Data-Generator/tree/main). This is required for the system to work correctly.

## I. Backend

### Requirements

* Python **3.10**
* MongoDB (local or cloud)
* `pip` for installing Python packages

### 1. Clone the Repository

```bash
git clone https://github.com/VarshithPawarHR/HPE-StoragePrediction
cd HPE-StoragePrediction/backend
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables

Create a `.env` file inside the `backend` folder: (refer `.env.example`)

```bash
TF_ENABLE_ONEDNN_OPTS=0
MONGO_URL=mongodb+srv://<username>:<password>@<cluster-url>/<database>
MONGO_DB=your_db_name
MONGO_COLLECTION=your_collection
```

### 4. Run the Backend

```bash
fastapi dev main.py
```

FastAPI will be live at: `http://127.0.0.1:8000/`

### Test API (Swagger UI)

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to manually test the APIs.

---

## II. Frontend

### Requirements

* Node.js **v18**
* A package manager (`npm` or `yarn`)

### 1. Clone the Repository

```bash
git clone https://github.com/VarshithPawarHR/HPE-Dashboard
cd HPE-Dashboard
```

### 2. Set Environment Variables

Create a `.env` file inside the HPE-Dashboard folder (refer `.env.example`):

```bash
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000/
MONGO_URL=mongodb+srv://<username>:<password>@<cluster-url>/<database>
MONGO_DB=your_db_name
MONGO_COLLECTION=your_collection
```

### 3. Install Dependencies

```bash
npm install
# or
yarn install
```

### 4. Run the Frontend

```bash
npm run dev
# or
yarn dev
```

Dashboard will be available at: `http://localhost:3000`

---

## Summary

* **Backend** runs on `http://127.0.0.1:8000`
* **Frontend** runs on `http://localhost:3000`
* Connected via REST APIs

Once both are running, navigate to the dashboard to view real-time storage forecasts.
