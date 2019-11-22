FROM tensorflow/tensorflow:nightly-gpu-py3 AS BASE

COPY requirements.txt /tmp/requirements.txt

RUN pip install -r /tmp/requirements.txt

COPY src /root/thesis


