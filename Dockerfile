FROM bioconductor/bioconductor_docker:RELEASE_3_22

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --break-system-packages -r /app/requirements.txt

COPY . /app

EXPOSE 10000

CMD ["sh", "-c", "python3 -m streamlit run adaptive_streamlit_app.py --server.address 0.0.0.0 --server.port ${PORT:-10000}"]
