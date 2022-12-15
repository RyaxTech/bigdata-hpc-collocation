FROM ubuntu:20.04


ENV PATH=/usr/local/spark/bin:/usr/local/spark/sbin:$PATH
# Port option should match what we set Dropbear to listen on
ENV SPARK_SSH_OPTS="-p 2222 -o StrictHostKeyChecking=no"
ENV SPARK_HOME=/usr/local/spark
ENV export PATH SPARK_SSH_OPTS SPARK_HOME
ENV SPARK_VERSION="3.1.2"

# ----------------------------------------------------
# Install useful prerequisites, and set python version
# ----------------------------------------------------
RUN apt-get update && apt-get upgrade -y &&\
    apt-get install -y software-properties-common curl &&\
    apt-get install -y python3 python3-dev &&\
    apt-get clean &&\
    # Make python3 the default
    update-alternatives --install /usr/bin/python python /usr/bin/python3 1 &&\
    # Upgrade pip to latest version
    curl -s https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py --force-reinstall && \
    rm get-pip.py

# -------------------------------
# Install our python dependencies
# -------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN /bin/bash -c '\
  mkdir -p /data; \
  apt-get update && apt-get install -y curl wget gzip \
        rsync openjdk-8-jre; \
  cd /usr/local; \
  wget https://archive.apache.org/dist/spark/spark-$SPARK_VERSION/spark-$SPARK_VERSION-bin-hadoop3.2.tgz; \
  gzip -d spark-$SPARK_VERSION-bin-hadoop3.2.tgz; \
  tar xf spark-$SPARK_VERSION-bin-hadoop3.2.tar; \
  mv spark-$SPARK_VERSION-bin-hadoop3.2 spark; \
  rm -f spark-$SPARK_VERSION-bin-hadoop3.2.tar; \
  apt-get install -y --no-install-recommends --allow-change-held-packages openssh-client dropbear; \
  sed -i -e "s@\(DROPBEAR_PORT=\).*@\12222@" /etc/default/dropbear; \
'

WORKDIR /data
ADD ./spark-entrypoint.sh /data/entrypoint.sh
ENTRYPOINT ["/data/entrypoint.sh"]
