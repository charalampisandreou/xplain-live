# xPlain Live: the Studio page + /api/bundle + /score, one image.
# Build:  docker build -t xplain-live:local .
# Run:    docker run --rm -p 8000:8000 xplain-live:local
FROM python:3.13-slim

ENV PYTHONPATH=src \
    PYTHONIOENCODING=utf-8 \
    PYTHONUTF8=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    XPLAIN_BUNDLE_DIR=/app/data

WORKDIR /app

COPY requirements-runtime.txt .
RUN pip install -r requirements-runtime.txt

# the app, the page (src/xplain/web) and the xPass model trained in this repo (xpass_v2 is loaded at startup)
COPY src/ src/
COPY models/xpass_v2/xpass_v2.joblib models/xpass_v2/feature_list.json models/xpass_v2/
# exactly one match bundle (data/ is git-ignored and otherwise stays out of the image)
COPY data/bundle_seed3001.json data/bundle_seed3001.json

# no secrets, keys or Azure credentials: serving needs none
RUN useradd --create-home --uid 10001 xplain
USER xplain

EXPOSE 8000
CMD ["uvicorn", "xplain.api:app", "--host", "0.0.0.0", "--port", "8000"]
