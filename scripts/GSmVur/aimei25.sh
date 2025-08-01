python cal_perstep_uncertainty_GSmVur.py \
    --data_path data/aime2025_test.json \
    --file_name amei25_gsmvur_4b_alpha0.7 \
    --gpus 1 \
    --aim_gpu 1 \
    --policy /apdcephfs_qy3/share_1443437/xinyuche/model/qwen3-4b \
    --critic /apdcephfs_qy3/share_1443437/xinyuche/model/genprm-1.5b \
    --scaling_rate 1 \
    --momentum_rate 0.7
python ./eval/math_verifier.py \
    --test_file ./res/amei25_gsmvur_4b_alpha0.7.json \
    --save_name amei25_gsmvur_4b_alpha0.7 \
    --aim_gpu 1
