FROM python:3.12-alpine

COPY application/ .

CMD ["python", "./HelloWorld.py"]