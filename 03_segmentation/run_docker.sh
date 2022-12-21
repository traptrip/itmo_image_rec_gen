docker build -t cv_labs . && docker system prune -f
docker run \
    -it \
    --rm \
    -p 8888:8888 \
    --gpus=all \
    --ipc=host \
    -v $PWD:/cv_labs/03_segmentation \
    --name cv_labs \
    cv_labs:latest 'jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root'
