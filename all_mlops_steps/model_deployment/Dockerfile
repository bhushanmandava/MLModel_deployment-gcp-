FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# List installed packages and their versions
RUN pip list

# Try to import wandb and print version
RUN python -c "import wandb; print(wandb.__version__)" || echo "Failed to import wandb"

COPY . .

ENV WANDB_API_KEY=da37b14ce73e93a3b20284def1a1b537024ed542
ENV WANDB_PROJECT=creadit-card-default-prediction_test-101

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9090"]