python3 ../ELP_SCC.py \
--dataset cityscapes \
--data_root "./dataset/ZSDE/city" \
--resize_feat \
--batch_size 16 \
--DKI_checkpoints "../DKI_weights/CS2Snow/DKI.pth" \
--Style_dir "../Styles/CS2Snow2/" \
--Style_dir2 "../Styles/CS2SnowSCC3/" \
--alpha_s 0.05 \
--l 20 \
--SCC \
--ELP \
--target_domain snow \
--gpu_id 0 \



