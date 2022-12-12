# 4.53 GB image, from https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-tensorrt
# could have used nvcr.io/nvidia/l4t-ml:r35.1.0-py3 from https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-ml which is 15.4 GB
FROM nvcr.io/nvidia/l4t-tensorrt:r8.4.1-runtime

WORKDIR /app

ARG DEBIAN_FRONTEND=noninteractive

# 4.53 ==> 4.9 GB
RUN apt update && apt install -y --no-install-recommends \
    build-essential python3-pip python3-dev cmake tzdata wget && \
    wget -q https://nvidia.box.com/shared/static/v59xkrnvederwewo2f1jtv6yurl92xso.whl -O onnxruntime_gpu-1.12.1-cp38-cp38-linux_aarch64.whl && \
    python3 -m pip install --upgrade --no-cache-dir pip wheel setuptools && \
    python3 -m pip install --no-cache-dir pillow onnx onnxruntime_gpu-1.12.1-cp38-cp38-linux_aarch64.whl && \
    rm -rf onnxruntime_gpu-1.12.1-cp38-cp38-linux_aarch64.whl && \
    apt remove -y --purge wget cmake build-essential python3-pip && \
    apt autoremove -y --purge && apt clean && \
    ln -fs /usr/share/zoneinfo/America/Chicago /etc/localtime && dpkg-reconfigure --frontend noninteractive tzdata

COPY Dockerfile .
COPY model.onnx .
COPY labels.txt .
COPY cvexport.manifest .
COPY onnxScore.py .

CMD [ "python3", "-u", "./onnxScore.py" ]