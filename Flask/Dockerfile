# our base image
FROM python:3-onbuild
# Dockerfile - this is a comment. Delete me if you want.

COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["app.py"]