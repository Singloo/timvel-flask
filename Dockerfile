FROM python:3.6-slim-stretch

RUN apt update
RUN apt install -y python3-dev gcc

# Install pytorch and fastai
RUN pip install torch_nightly -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
RUN pip --no-cache-dir install fastai

# Install librarys
RUN pip install flask

ADD . .

EXPOSE 5000

# Start the server
CMD ["python", "app.py"]