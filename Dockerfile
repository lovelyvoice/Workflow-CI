FROM python:3.12-slim 
WORKDIR /app 
RUN pip install mlflow==2.21.0 scikit-learn==1.5.2 pandas==2.2.3 numpy==1.26.4 matplotlib==3.9.2 seaborn==0.13.2 
COPY mlruns/ /app/mlruns/ 
ENV MLFLOW_TRACKING_URI=file:///app/mlruns 
EXPOSE 5000 
CMD ["mlflow", "models", "serve", "--host", "0.0.0.0", "--port", "5000", "--env-manager", "local", "--model-uri", "mlruns/0"] 
