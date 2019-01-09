FROM sberbank/python
ADD automl ./app
ADD entrypoint.sh ./app/entrypoint.sh
#ADD calculate_rating.sh ./app/calculate_rating.sh
RUN mkdir /models
CMD ./app/entrypoint.sh
