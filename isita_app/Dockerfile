# Copyright (c) 2018 OPTUM All rights reserved.
FROM continuumio/anaconda3

EXPOSE 5000

#set working directory
#USER = root
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

#add requirements.txt
ADD ./requirements.txt /usr/src/app/requirements.txt

#install requirements
RUN pip install -r requirements.txt

RUN mkdir -p /usr/src/app/model/
COPY model/ /usr/src/app/model/

RUN mkdir -p /usr/src/app/static/
COPY static/ /usr/src/app/static/

RUN mkdir -p /usr/src/app/templates/
COPY templates/ /usr/src/app/templates/

RUN mkdir -p /usr/src/app/UPLOAD/
COPY UPLOAD/ /usr/src/app/UPLOAD/


RUN chown -R 1001:0 /usr/ && \
    chmod -R g+wrx /usr/

USER 1001

#Run server.py code
ENTRYPOINT ["python"]
CMD ["server.py"]
