FROM python:3.8 as poetry2requirements
COPY pyproject.toml poetry.lock README.md /
ENV POETRY_HOME=/etc/poetry
RUN pip3 install poetry==1.3.2
RUN python3 -m poetry export --without-hashes -f requirements.txt \
    | grep -v "torch=" \
    > /Requirements.txt


FROM nvcr.io/nvidia/pytorch:22.12-py3

ENV MPLCONFIGDIR=/tmp

COPY --from=poetry2requirements /Requirements.txt /tmp

WORKDIR /app

COPY salt /app/salt
COPY models /app/models

RUN cat /tmp/Requirements.txt

# Install app dependencies
RUN pip3 install -U pip && \
    pip3 install -r /tmp/Requirements.txt && \
    rm /tmp/Requirements.txt
