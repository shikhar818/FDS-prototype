# Hotel Cancelation Prediction and Revenue Forecasting
Hotels are prone to last-second cancellations, which they cant afford as they are already running
on minimal margins. It is crucial for them to be able to predict whether or not a customer would
cancel and how much does the hotel stand to make from it. To maximise profits, hotels should
focus on a certain customer type, that stands to make them the most profit. Understanding and
catering to a certain customer base can not only increase profits but also help reduce operational
costs. This report intends to help hotel administration and ownership to make informed decisions
regarding bookings and their target customers


## Installation and Setup

Follow the steps below to set up and run the project locally.

### 1. Create a Conda Environment

Create a new Conda environment with Python 3.11:

```bash
conda create -n hotel-revenue python=3.11 -y
```

Activate the environment:

```bash
conda activate hotel-revenue
```

### 2. Install PyTorch

Install PyTorch inside the Conda environment.

For CUDA-enabled systems (NVIDIA GPU):

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

For CPU-only systems:

```bash
pip install torch torchvision torchaudio
```

Verify the installation:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### 3. Install UV

Install UV, the Python package and project management tool:

```bash
pip install uv
```

Verify the installation:

```bash
uv --version
```

### 4. Install Project Dependencies

Navigate to the project root directory (the directory containing `requirements.txt`).

Install all required dependencies:

```bash
pip install -r requirements.txt
```

### 5. Run the Backend (FastAPI)

Start the FastAPI backend using Uvicorn:

```bash
uvicorn app.main:app --reload
```

The backend will be available at:

- **API:** http://127.0.0.1:8000
- **Swagger API Documentation:** http://127.0.0.1:8000/docs

### 6. Run the Frontend (Streamlit)

Open a **separate terminal**, activate the same Conda environment, and navigate to the project root directory.

```bash
conda activate hotel-revenue
```

Start the Streamlit dashboard:

```bash
streamlit run app/dashboard.py
```

The frontend will be available at:

- **Dashboard:** http://localhost:8501

### 7. Running the Application

To run the complete application, keep both terminals running:

| Component | Command |
|---|---|
| Backend | `uvicorn app.main:app --reload` |
| Frontend | `streamlit run app/dashboard.py` |

**Note:** Ensure the backend is running before using the Streamlit dashboard, as the frontend communicates with the FastAPI backend for predictions.