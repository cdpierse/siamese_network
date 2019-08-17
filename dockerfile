FROM continuumio/anaconda3
# RUN conda update conda
# COPY /test.py .
RUN echo "conda create env -f environment.yml" > ~/.bashrc
RUN echo "source activate siamese_network" > ~/.bashrc
# RUN python ./test.py