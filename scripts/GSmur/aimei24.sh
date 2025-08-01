python cal_perstep_uncertainty_GSmur.py \
    --data_path data/aime2024_test.json \
    --file_name amei24_gsmur_1.8b \
    --gpus 1 \
    --aim_gpu 0 \
    --policy /apdcephfs_qy3/share_1443437/xinyuche/model/qwen3-1.8b \
    --critic /apdcephfs_qy3/share_1443437/xinyuche/model/genprm-1.5b \
    --scaling_rate 0.9

python ./eval/math_verifier.py \
    --test_file ./res/amei24_gsmur_1.8b.json \
    --save_name amei24_gsmur_1.8b \
    --aim_gpu 0
