python3 ../DKI_training.py \
--dataset cityscapes \
--data_root "./dataset/ZSDE/city" \
--resize_feat \
--batch_size 64 \
--lambda_s 0.1 \
--DKI_save_dir "../DKI_weights/CS2Night/" \
--target_domain night \
--training_epochs 200 \
--gpu_id 2 \
--train