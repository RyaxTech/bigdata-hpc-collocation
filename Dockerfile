FROM ubuntu:16.04


ENV PATH=/usr/local/spark/bin:/usr/local/spark/sbin:$PATH
# Port option should match what we set Dropbear to listen on
ENV SPARK_SSH_OPTS="-p 2222 -o StrictHostKeyChecking=no"
ENV SPARK_HOME=/usr/local/spark
ENV export PATH SPARK_SSH_OPTS SPARK_HOME
ENV SPARK_VERSION="2.4.5"

RUN /bin/bash -c '\
  mkdir -p /data; \
  apt-get update && apt-get install -y curl wget gzip \
        rsync openjdk-8-jre python2.7; \
  ln -s /usr/bin/python2.7 /usr/bin/python; \
  cd /usr/local; \
  wget https://archive.apache.org/dist/spark/spark-$SPARK_VERSION/spark-$SPARK_VERSION-bin-hadoop2.7.tgz; \
  gzip -d spark-$SPARK_VERSION-bin-hadoop2.7.tgz; \
  tar xf spark-$SPARK_VERSION-bin-hadoop2.7.tar; \
  mv spark-$SPARK_VERSION-bin-hadoop2.7 spark; \
  rm -f spark-$SPARK_VERSION-bin-hadoop2.7.tar; \
  apt-get install -y --no-install-recommends --allow-change-held-packages openssh-client dropbear; \
  sed -i -e "s@\(DROPBEAR_PORT=\).*@\12222@" /etc/default/dropbear; \
'

ADD ./spark-entrypoint.sh /data/entrypoint.sh
ENTRYPOINT ["/data/entrypoint.sh"]
