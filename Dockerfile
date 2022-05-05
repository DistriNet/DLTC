FROM tensorflow/tensorflow:2.3.1-gpu-jupyter
COPY exp/[^data]* /app/exp/
RUN pip3 install -r /app/exp/requirements.txt
EXPOSE 8888
WORKDIR /app/exp/
CMD jupyter notebook . --allow-root --no-browser --ip 0.0.0.0 
