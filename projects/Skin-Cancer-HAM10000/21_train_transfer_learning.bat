rem 1_train.bat
python ../../EfficientNetV2ModelTrainer.py ^
  --model_dir=./models_tl ^
  --model_name=efficientnetv2-m  ^
  --eval_dir=./eval_tl ^
  --data_generator_config=./data_generator.config ^
  --ckpt_dir=../../efficientnetv2-m/model ^
  --optimizer=rmsprop ^
  --image_size=384 ^
  --eval_image_size=480 ^
  --data_dir=./Resampled_HAM10000/Training ^
  --data_augmentation=True ^
  --valid_data_augmentation=True ^
  --fine_tuning=False ^
  --monitor=val_loss ^
  --learning_rate=0.0001 ^
  --trainable_layers_ratio=0.4 ^
  --dropout_rate=0.3 ^
  --num_epochs=50 ^
  --batch_size=4 ^
  --patience=10 ^
  --debug=True  


