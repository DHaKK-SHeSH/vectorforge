# VectorForge search service.
#   docker build -t vectorforge .
#   docker run -p 8000:8000 -v "$PWD:/data" -e VF_INDEX=/data/index.vf.npz vectorforge
# (build the index file first: python scripts/build_index.py --out index.vf.npz)
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY vectorforge ./vectorforge
RUN pip install --no-cache-dir ".[service]"
ENV VF_INDEX=/data/index.vf.npz
EXPOSE 8000
CMD ["uvicorn", "vectorforge.service:app", "--host", "0.0.0.0", "--port", "8000"]
