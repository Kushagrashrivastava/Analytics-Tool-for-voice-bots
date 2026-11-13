FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[web]"
COPY app ./app
COPY .streamlit ./.streamlit
COPY data ./data
EXPOSE 8501
# Default: dashboard. CLI: docker run --env GROQ_API_KEY -v $PWD:/w aster-vale aster-vale run --input /w/in.jsonl --output /w/out.jsonl
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
