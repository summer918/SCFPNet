CUDA_VISIBLE_DEVICES=0,1 python3 fsod_train_net.py --num-gpus 2 --dist-url auto \
	--config-file configs/fsod/two_branch_training_coco_pvt_v2_b2_li.yaml SOLVER.IMS_PER_BATCH 2 2>&1 | tee log/two_branch_training_coco_pvt_v2_b2_li.txt
