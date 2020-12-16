FROM jupyter/datascience-notebook
WORKDIR /capstone
COPY README.md .
CMD ["jupyter", "notebook"]

FROM python
WORKDIR /capstone
COPY requirements.txt .
RUN "pip install -r requirements.txt"

