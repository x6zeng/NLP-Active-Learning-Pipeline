ARG BASE_CONTAINER=ucsdets/datahub-base-notebook:2022.3-stable

FROM $BASE_CONTAINER

COPY . .

USER root

RUN pip install -r requirements.txt

CMD ["/bin/bash"]