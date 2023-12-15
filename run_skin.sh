#!/bin/bash
declare -a splits=(1 2 3 4 5 6 7 8 9 10)
#declare -a splits=(1 2)
#declare -a lrs=("0.0001")
#declare -a wds=("0.0001" "0.1")
#declare -a bias_factor=("80")
declare -a bias_factor=("52" "56" "60" "64" "68" "72" "76" "80")
#declare -a gen_adjusts=(0)
square_size=63
GPU=$1
cnoise=80

for split in "${splits[@]}"; do
    #for lr in "${lrs[@]}"; do
        #for wd in "${wds[@]}"; do
#            for ga in "${gen_adjusts[@]}"; do
                for bf in "${bias_factor[@]}"; do
                    CUDA_VISIBLE_DEVICES=$GPU nohup python run_expt_comet.py with cfg.batch_size=128 cfg.shift_type=confounder cfg.dataset=Biased cfg.model=resnet18 cfg.weight_decay=0.0001 cfg.lr=0.0001 cfg.n_epochs=100 cfg.save_step=2 cfg.save_best=True cfg.save_last=True cfg.reweight_groups=True cfg.robust=True cfg.alpha=0.01 cfg.gamma=0.1 cfg.generalization_adjustment=0 cfg.exp_desc=GroupDRO cfg.square_size=${square_size} cfg.bias_factor=${bf} cfg.train_from_scratch=False cfg.random_square_position=True cfg.color_noise=${cnoise} cfg.patience=100 cfg.seed=${split} cfg.exp_name=groupdro_final_pretrained_skin_squared_position_cnoise${cnoise}_bf${bf}_lr00001_wd00001_split${split}_adjustment0
                    CUDA_VISIBLE_DEVICES=$GPU nohup python run_expt_comet.py with cfg.batch_size=128 cfg.shift_type=confounder cfg.dataset=Biased cfg.model=resnet18 cfg.weight_decay=0.1 cfg.lr=0.0001 cfg.n_epochs=100 cfg.save_step=2 cfg.save_best=True cfg.save_last=True cfg.reweight_groups=False cfg.robust=False cfg.alpha=0.01 cfg.gamma=0.1 cfg.generalization_adjustment=0 cfg.exp_desc=Baseline cfg.square_size=${square_size} cfg.bias_factor=${bf} cfg.train_from_scratch=False cfg.random_square_position=True cfg.color_noise=${cnoise} cfg.patience=100 cfg.seed=${split} cfg.exp_name=baseline_final_pretrained_skin_squared_position_cnoise${cnoise}_bf${bf}_lr00001_wd01_split${split}_adjustment0

                done
#            done
        #done
    #done
done



