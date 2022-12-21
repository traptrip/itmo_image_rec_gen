docker build -t cv_lab . && \
docker system prune -f && \
docker run \
    -it \
    --rm \
    -p 8889:8889 \
    --gpus=all \
    --ipc=host \
    -v $PWD:/cv_lab \
    --name cv_labs \
    cv_lab:latest 'jupyter notebook --ip=0.0.0.0 --port=8889 --no-browser --allow-root'
