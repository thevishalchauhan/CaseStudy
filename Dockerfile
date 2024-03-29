FROM python
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8080
CMD streamlit run main.py --server.port 8080