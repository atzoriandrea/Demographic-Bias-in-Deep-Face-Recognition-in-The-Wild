mkdir 'log'
python train.py \
    --data_root_train '' \
    --train_file '' \
    --backbone_type 'ResNet' \
    --backbone_conf_file '../model_conf_files/backbone_conf/backbone_conf.yaml' \
    --head_type 'MagFace' \
    --head_conf_file '../model_conf_files/head_conf/head_conf.yaml' \
    --lr 0.01 \
    --out_dir '' \
    --epoches 85 \
    --step '5,12,22,35,48' \
    --print_freq 200 \
    --save_freq 3000 \
    --batch_size 32 \
    --log_dir 'log' \
    --cuda True \
    --saveall True\
    --pretrain_model ''\
    --resume_from_epoch 35 \
    --tensorboardx_logdir 'mv-hrnet' \
    2>&1 | tee log/log.log
