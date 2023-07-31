FROM python:3.10.8
WORKDIR /bot
RUN pip install --upgrade pip
RUN pip install pandas
RUN pip install tensorflow
RUN pip install tflearn
RUN pip install numpy
RUN pip install nltk
RUN pip install discord
COPY . /bot/
CMD python Isc_discord.py

