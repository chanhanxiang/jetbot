# From the tensorflow/models/research/ directory
PIPELINE_CONFIG_PATH=/home/ubuntu/project4b/pipeline.config
MODEL_DIR=/home/ubuntu/project4b/models/model
TRAIN_CHECKPOINT_DIR=/home/ubuntu/project4b/models/model
EXPORT_DIR=/home/ubuntu/project4b/models/export_model

python /home/ubuntu/git/models/research/object_detection/exporter_main_v2.py \
    --input_type image_tensor \
    --pipeline_config_path ${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_dir ${TRAIN_CHECKPOINT_DIR} \
    --output_directory ${EXPORT_DIR}
    
