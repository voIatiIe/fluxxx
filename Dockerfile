FROM --platform=amd64 ubuntu:20.04

RUN apt-get update
RUN apt-get install -y gcc g++ make cmake curl

RUN curl -sL https://github.com/go-task/task/releases/download/v3.34.1/task_linux_amd64.deb -o task.deb \
    && dpkg -i task.deb \
    && rm task.deb

WORKDIR /app

COPY . /app

CMD ["/bin/bash"]
