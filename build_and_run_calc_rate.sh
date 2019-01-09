docker build . -t sber_mse &&
docker run\
    -v "$(dirname $(pwd))"/sdsj2018_automl_check_datasets:/data \
    -v "$(pwd)"/models:/models \
    sber_mse app/calculate_rating.sh
