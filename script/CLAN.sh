/data/liuhaofeng/miniconda3/envs/python3.6/bin/python /data/liuhaofeng/oumingyang/eye_cataract/2023Fall/train.py \
--dataroot /data/liuhaofeng/oumingyang/eye_cataract/dataset/CADIS-2/ \
--targetroot /data/liuhaofeng/oumingyang/eye_cataract/dataset/cataract-1/ \
--name CADIS-2_insegcat-1_CLAN_221013 \
--model CLAN \
--netG CLAN \
--netD CLAN \
--dataset_mode da \
--load_size 480 \
--crop_size 200 \
--preprocess scale_width \
--output_nc 36 \
--batch_size 5 \
--lr 0.00025 \
--D_lr 0.0001 \
--n_epochs 5 \
--n_epochs_decay 100 \
--display_freq 20 \
--display_port 8089 \
--gpu_ids 6