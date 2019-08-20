# FROM tensorflow/tensorflow:devel-py3
# COPY /requirements.txt .
# RUN pip install -r requirements.txt

FROM tensorflow/tensorflow:custom-op-ubuntu16

RUN apt-get update -y && \
    apt-get install -y python3-pip python3-dev

COPY ./requirements.txt /requirements.txt

WORKDIR /

RUN pip3 install -r requirements.txt

COPY . /

ENTRYPOINT [ "python3" ]