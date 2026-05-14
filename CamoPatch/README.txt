Example use:

pip install -r requirements.txt

python ConCamoPatch.py --model 1 --model_source bcos --image_dir ./8.JPEG --true_label 8 --save_directory 8 // Sequential CamoPatch on uploaded/local B-cos ResNet-50 weights

python ConCamoPatch.py --model 1 --model_source bcos --image_dir ./8.JPEG --true_label 8 --save_directory 8_linf --linf 8/255 // Sequential CamoPatch with an L-infinity bound

python ConCamoPatch.py --model 1 --model_source bcos --image_dir ./8.JPEG --true_label 8 --save_directory 8_fast --device cuda --batch_size 8 --trace_every 0 // Faster batched patch-candidate variant

Outputs include .npy, *_adversary.png, and *_patch.png.

Use --model_source torchvision to run the original torchvision ResNet-50 path. It loads offline weights from weights/torchvision-imagenet or an attached Kaggle dataset containing torchvision-imagenet/.

CSV sweep:
./run_resnet50_linf_sweep.sh ../used_images_500_1.csv results/camopatch_resnet50 --device cuda --batch_size 8 --trace_every 0

Default patch size is 16x16. Override with --s if needed.
