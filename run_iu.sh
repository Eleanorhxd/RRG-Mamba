python main.py --exp_name RRG-Mamba --label_path ./files/iu_xray/labels.json --dataset_name iu_xray --max_seq_length 60 --threshold 3 --batch_size 32 --epochs 100 --lr_ve 1e-3 --lr_ed 2e-3 --save_dir ../results/iu_xray --ve_name densenet121 --ed_name r2gen --cfg ./configs/swin_tiny_patch4_window7_224.yaml --early_stop 50 --weight_decay 5e-5 --optim Adam 
--decay_epochs 50 --warmup_epochs 0 --warmup_lr 1e-4 --lr_scheduler step --decay_rate 0.8 --seed 9233 --addcls  --cls_w 1 --fbl --attn_cam --attn_method max --topk 0.25 --layer_id 2 --mse_w 0.15 

