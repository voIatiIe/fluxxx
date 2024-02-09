FROM --platform=amd64 ubuntu:20.04

RUN apt-get update
RUN apt-get install -y gcc g++ make cmake curl unzip

RUN curl -sL https://github.com/go-task/task/releases/download/v3.34.1/task_linux_amd64.deb -o task.deb \
    && dpkg -i task.deb \
    && rm task.deb

RUN curl -sL https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip -o libtorch.zip \
    && unzip libtorch.zip -d /opt \
    && rm libtorch.zip

WORKDIR /app
COPY . /app

CMD ["/bin/bash"]
