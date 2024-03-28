FROM python:3.9 as poetry2requirements
COPY pyproject.toml poetry.lock /
ENV POETRY_HOME=/etc/poetry
RUN pip3 install poetry
RUN python3 -m poetry export --without-hashes -f requirements.txt \
    | grep -v "torch=" \
    > /Requirements.txt


FROM nvcr.io/nvidia/pytorch:22.12-py3

# Install app dependencies
COPY --from=poetry2requirements /Requirements.txt /tmp
RUN pip3 install -U pip && \
    pip3 install -r /tmp/Requirements.txt && \
    rm /tmp/Requirements.txt
