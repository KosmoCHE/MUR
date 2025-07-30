python cal_perstep_uncertainty_GSref.py \
    --data_path data/aime2025_test.json \
    --file_name amei25_gsref \
    --gpus 1 \
    --aim_gpu 1 \
    --policy /apdcephfs_qy3/share_1443437/xinyuche/model/qwen3-1.8b \
    --critic /apdcephfs_qy3/share_1443437/xinyuche/model/genprm-1.5b \
    --scaling_rate 1
python ./eval/math_verifier.py \
    --test_file ./res/amei25_gsref.json \
    --save_name amei25_gsref \
    --aim_gpu 1
