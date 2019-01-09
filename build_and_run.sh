arg=$1
if [[ ${arg: -1} = 'c' ]]
then
    mode="classification"
else
    mode="regression"
fi

docker build . -t sber_mse &&
docker run\
    -v "$(dirname $(pwd))"/sdsj2018_automl_check_datasets:/data \
    -v "$(pwd)"/models:/models \
    sber_mse app/entrypoint.sh $mode $1
