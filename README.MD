## ACP-demo
Due to size limitations, the training and testing datasets need to be downloaded online(https://paddleocr.bj.bcebos.com/dataset/ctw1500.zip). In addition, we provide extra predictions for individual images here. 


### Train
python tools/train.py --config_file ./config/ctw1500_ACP_FPN_DBhead_polyLR.yaml

### Predict the entire folder

python tools/predict.py --model_path [Model location]  --input_folder [Enter folder path] --output_folder [Output path] --show --polygon --thre [threshold(0.55)] --save_result
For instance：
python tools/predict.py --model_path ./output/model_best.pth  --input_folder ./datasets/datasets/ctw1500/imgs/test --output_folder result --show --polygon --thre 0.55 --save_result


### Predicting a single image
For instance：
python tools/predict.py --model_path ./output/model_best.pth  --input_image ./datasets/CWT1500/1123.jpg --output_folder result --show --polygon --thre 0.55 --save_result


### Environment configuration
cuda：10.2
anyconfig==0.9.10
numpy==1.22.3
opencv-python==4.9.0.80
scikit-image==0.21.0
tensorboard==2.14.0
tqdm==4.65.2
torch==1.8
torchvision==0.9

