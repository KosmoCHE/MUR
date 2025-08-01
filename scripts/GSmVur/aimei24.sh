python cal_perstep_uncertainty_GSmVur.py \
    --data_path data/aime2024_test.json \
    --file_name amei24_gsmvur \
    --gpus 1 \
    --aim_gpu 0 \
    --policy /apdcephfs_qy3/share_1443437/xinyuche/model/qwen3-1.8b \
    --critic /apdcephfs_qy3/share_1443437/xinyuche/model/genprm-1.5b \
    --scaling_rate 1 

python ./eval/math_verifier.py \
    --test_file ./res/amei24_gsmvur.json \
    --save_name amei24_gsmvur \
    --aim_gpu 0
