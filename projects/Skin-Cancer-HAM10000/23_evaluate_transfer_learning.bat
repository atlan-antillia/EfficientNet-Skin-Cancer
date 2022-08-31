rem 3_evaluate.bat
python ../../EfficientNetV2Evaluator.py ^
  --model_name=efficientnetv2-m  ^
  --model_dir=./models_tl ^
  --data_dir=./Resampled_HAM10000/Testing ^
  --evaluation_dir=./evaluation_tl ^
  --fine_tuning=False ^
  --trainable_layers_ratio=0.4 ^
  --dropout_rate=0.3 ^
  --eval_image_size=480 ^
  --mixed_precision=True ^
  --debug=False 
 