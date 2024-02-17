# From the tensorflow/models/research/ directory
export CUDA_VISIBLE_DEVICES=""
PIPELINE_CONFIG_PATH=/home/ubuntu/project4b/pipeline.config
MODEL_DIR=/home/ubuntu/project4b/models/model
CHECKPOINT_DIR=${MODEL_DIR}
python /home/ubuntu/git/models/research/object_detection/model_main_tf2.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --checkpoint_dir=${CHECKPOINT_DIR} \
    --alsologtostderr
