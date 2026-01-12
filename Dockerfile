FROM python:3.11-slim

WORKDIR /app


RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*


COPY . .


RUN pip install -e .


EXPOSE 8000


CMD ["python", "-m", "excel_agent.main", "serve"]