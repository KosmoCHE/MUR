python cal_perstep_uncertainty_naive.py \
    --data_path data/aime2024_test.json \
    --file_name amei24_naive_4b \
    --gpus 1 \
    --aim_gpu 1 \
    --policy /apdcephfs_qy3/share_1443437/xinyuche/model/qwen3-4b

python ./eval/math_verifier.py \
    --test_file ./res/amei24_naive_4b.json \
    --save_name amei24_naive_4b \
    --aim_gpu 0
