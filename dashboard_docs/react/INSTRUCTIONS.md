# Instructions: Running the React Dashboard

To run the React dashboard, you need to start two processes in separate terminal windows: the Python data API and the React user interface.

## Step 1: Start the FastAPI Backend
The backend processes your MLflow and Optuna data on the fly so the dashboard can display it. 
Open a terminal in the root folder (`d:\Projects\ml-cost-tracker`) and run:

```bash
# Using the existing virtual environment:
.\.venv\Scripts\python.exe src\api\main.py
```
*The backend will spin up and listen on `http://127.0.0.1:8000`.*

---

## Step 2: Start the React Frontend
The frontend renders the polished UI.
Open a **second, separate terminal window** in the root folder and run:

```bash
cd frontend
npm run dev
```
*The Vite engine will compile the UI in a few seconds.*

---

## Step 3: View the Dashboard
Once both servers are running, open your favorite web browser and navigate to:
👉 **[http://localhost:5173](http://localhost:5173)**

> **Note:** If you ever run new Optuna experiments or run the feature builder, you don't need to restart the servers! Just refresh your web browser and the backend will pull the latest data automatically.
