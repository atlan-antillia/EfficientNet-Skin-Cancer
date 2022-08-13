<h2>EfficientNet-Skin-Cancer (Updated: 2022/08/13)</h2>
<a href="#1">1 EfficientNetV2 Skin Cancer HAM10000 Classification </a><br>
<a href="#1.1">1.1 Clone repository</a><br>
<a href="#1.2">1.2 Install Python packages</a><br>
<a href="#2">2 Python classes for Skin Cancer Detection</a><br>
<a href="#3">3 Pretrained model</a><br>
<a href="#4">4 Train</a><br>
<a href="#4.1">4.1 Train script</a><br>
<a href="#4.2">4.2 Training result</a><br>
<a href="#5">5 Inference</a><br>
<a href="#5.1">5.1 Inference script</a><br>
<a href="#5.2">5.2 Sample test images</a><br>
<a href="#5.3">5.3 Inference result</a><br>
<a href="#6">6 Evaluation</a><br>
<a href="#6.1">6.1 Evaluation script</a><br>
<a href="#6.2">6.2 Evaluation result</a><br>

<h2>
<a id="1">1 EfficientNetV2 Skin Cancer HAM10000 Classification</a>
</h2>

 This is an experimental Skin Cancer HAM10000 Classification project based on <b>efficientnetv2</b> in <a href="https://github.com/google/automl">Brain AutoML</a>
 The original Skin Cancer HAM10000 dataset has been taken from the following web site:<br>
<b>The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions</b><br>

https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T 
<br>
However, we have used 
<a href="https://drive.google.com/file/d/1OqRiuFArflpw-8Anm2UV4EdyfS77ANTA/view?usp=sharing">Resampled HAM10000</a>
 dataset created by <a href="https://github.com/martian-antillia/ImageDatasetResampler">ImageDatasetResampler</a>.
<br>
<br>
<br>We use python 3.8 and tensorflow 2.8.0 environment on Windows 11.<br>
  
<h3>
<a id="1.1">1.1 Clone repository</a>
</h3>
 Please run the following command in your working directory:<br>
<pre>
git clone https://github.com/atlan-antillia/EfficientNet-Skin-Cancer.git
</pre>
You will have the following directory tree:<br>
<pre>
.
├─asset
└─projects
    └─Skin-Cancer-HAM10000
        ├─eval
        ├─evaluation
        ├─inference        
        └─test
</pre>
Please download the dataset from the following web site:
<br>
<a href="https://drive.google.com/file/d/1OqRiuFArflpw-8Anm2UV4EdyfS77ANTA/view?usp=sharing">Resampled_HAM10000.zip</a> 
<br>
, expand it, and place the dataset <b>Resampled_HAM10000</b> folder under <b>Skin-Cancer-HAM10000</b> folder.<br>

<pre>
.
├─asset
├─efficientnetv2-m
└─projects
    └─Skin-Cancer-HAM10000
        ├─eval
        ├─evaluation
        ├─inference
        ├─Resampled_HAM10000
        │  ├─Testing
        │  │  ├─akiec
        │  │  ├─bcc
        │  │  ├─bkl
        │  │  ├─df
        │  │  ├─mel
        │  │  ├─nv
        │  │  └─vasc
        │  └─Training
        │      ├─akiec
        │      ├─bcc
        │      ├─bkl
        │      ├─df
        │      ├─mel
        │      ├─nv
        │      └─vasc
        └─test
     　...
</pre>
<br> 
<h3>
<a id="#1.2">1.2 Install Python packages</a>
</h3>
<br>
Please run the following commnad to install Python packages for this project.<br>
<pre>
pip install -r requirements.txt
</pre>
<br>

<h2>
<a id="2">2 Python classes for Skin Cancer HAM10000 Classification</a>
</h2>
We have defined the following python classes to implement our Skin Cancer HAM10000 Classification.<br>
<li>
<a href="./ClassificationReportWriter.py">ClassificationReportWriter</a>
</li>
<li>
<a href="./ConfusionMatrix.py">ConfusionMatrix</a>
</li>
<li>
<a href="./CustomDataset.py">CustomDataset</a>
</li>
<li>
<a href="./EpochChangeCallback.py">EpochChangeCallback</a>
</li>
<li>
<a href="./EfficientNetV2Evaluator.py">EfficientNetV2Evaluator</a>
</li>
<li>
<a href="./EfficientNetV2Inferencer.py">EfficientNetV2Inferencer</a>
</li>
<li>
<a href="./EfficientNetV2ModelTrainer.py">EfficientNetV2ModelTrainer</a>
</li>
<li>
<a href="./FineTuningModel.py">FineTuningModel</a>
</li>

<li>
<a href="./TestDataset.py">TestDataset</a>
</li>

<h2>
<a id="3">3 Pretrained model</a>
</h2>
 We have used pretrained <b>efficientnetv2-m</b> to train Skin-Cancer-HAM10000 Model.
Please download the pretrained checkpoint file from <a href="https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/v2/efficientnetv2-m.tgz">efficientnetv2-m.tgz</a>, expand it, and place the model under our top repository.

<pre>
.
├─asset
├─efficientnetv2-m
└─projects
    └─Skin-Cancer-HAM10000
        └─test
     　...
</pre>

<h2>
<a id="4">4 Train</a>

</h2>
<h3>
<a id="4.1">4.1 Train script</a>
</h3>
Please run the following bat file to train our Skin Cancer HAM10000 efficientnetv2 model by using
<b>Resampled_HAM10000/Training</b>.
<pre>
./1_train.bat
</pre>
<pre>
rem 1_train.bat
python ../../EfficientNetV2ModelTrainer.py ^
  --model_dir=./models ^
  --model_name=efficientnetv2-m  ^
  --data_generator_config=./data_generator.config ^
  --ckpt_dir=../../efficientnetv2-m/model ^
  --optimizer=adam ^
  --image_size=384 ^
  --eval_image_size=480 ^
  --data_dir=./Resampled_HAM10000/Training ^
  --model_dir=./models ^
  --data_augmentation=True ^
  --fine_tuning=True ^
  --monitor=val_loss ^
  --learning_rate=0.0001 ^
  --trainable_layers_ratio=0.4 ^
  --num_epochs=50 ^
  --batch_size=4 ^
  --patience=10 ^
  --debug=True  

</pre>
, where data_generator.config is the following:<br>
<pre>
; data_generation.config

[training]
validation_split   = 0.2
featurewise_center = True
samplewise_center  = False
featurewise_std_normalization=True
samplewise_std_normalization =False
zca_whitening                =False
rotation_range     = 30
horizontal_flip    = True
vertical_flip      = True
 
width_shift_range  = 0.1
height_shift_range = 0.1
shear_range        = 0.1
zoom_range         = [0.5, 1.5]
;zoom_range         = 0.2
data_format        = "channels_last"

[validation]
validation_split   = 0.2
featurewise_center = True
samplewise_center  = False
featurewise_std_normalization=True
samplewise_std_normalization =False
zca_whitening                =False
rotation_range     = 30
horizontal_flip    = True
vertical_flip      = True
width_shift_range  = 0.1
height_shift_range = 0.1
shear_range        = 0.1
zoom_range         = [0.5, 1.5]
;zoom_range         = 0.2
data_format        = "channels_last"
</pre>

<h3>
<a id="4.2">4.2 Training result</a>
</h3>

This will generate a <b>best_model.h5</b> in the models folder specified by --model_dir parameter.<br>
Furthermore, it will generate a <a href="./projects/Skin-Cancer-HAM10000/eval/train_accuracies.csv">train_accuracies</a>
and <a href="./projects/Skin-Cancer-HAM10000/eval/train_losses.csv">train_losses</a> files
<br>
Training console output:<br>
<img src="./asset/Skin_Cancer_train_console_output_at_epoch_28_0813.png" width="740" height="auto"><br>
<br>
Train_accuracies:<br>
<img src="./asset/Skin_Cancer_accuracies_at_epoch_28_0813.png" width="740" height="auto"><br>

<br>
Train_losses:<br>
<img src="./asset/Skin_Cancer_train_losses_at_epoch_28_0813.png" width="740" height="auto"><br>

<br>
<h2>
<a id="5">5 Inference</a>
</h2>
<h3>
<a id="5.1">5.1 Inference script</a>
</h3>
Please run the following bat file to infer the skin cancer lesions in test images by the model generated by the above train command.<br>
<pre>
./2_inference.bat
</pre>
<pre>
rem 2_inference.bat
python ../../EfficientNetV2Inferencer.py ^
  --model_name=efficientnetv2-m  ^
  --model_dir=./models ^
  --fine_tuning=True ^
  --trainable_layers_ratio=0.4 ^
  --image_path=./test/*.jpg ^
  --eval_image_size=480 ^
  --label_map=./label_map.txt ^
  --mixed_precision=True ^
  --infer_dir=./inference ^
  --debug=False 
</pre>
<br>
label_map.txt:
<pre>
akiec
bcc
bkl
df
mel
nv
vasc
</pre>
<br>
<h3>
<a id="5.2">5.2 Sample test images</a>
</h3>

Sample test images generated by <a href="./projects/Skin-Cancer-HAM10000/create_test_dataset.py">create_test_dataset.py</a> 
from <a href="./projects/Skin-Cancer-HAM10000/Resampled_HAM10000/Testing">Resampled_HAM10000/Testing</a>.
<br>
akiec<br>
<img src="./projects/Skin-Cancer-HAM10000/test/akiec___ISIC_0029567_102.jpg"  width="400" height="auto"><br><br>
bcc
<br>
<img src="./projects/Skin-Cancer-HAM10000/test/bcc___ISIC_0029391_103.jpg"  width="400" height="auto"><br><br>
bkl<br>
<img src="./projects/Skin-Cancer-HAM10000/test/bkl___ISIC_0030705_104.jpg"  width="400" height="auto"><br><br>
df<br>
<img src="./projects/Skin-Cancer-HAM10000/test/df___ISIC_0029760_109.jpg"  width="400" height="auto"><br><br>
mel<br>
<img src="./projects/Skin-Cancer-HAM10000/test/mel___ISIC_0030060_108.jpg"  width="400" height="auto"><br><br>
nv<br>
<img src="./projects/Skin-Cancer-HAM10000/test/nv___ISIC_0029350_106.jpg"  width="400" height="auto"><br><br>
vasc<br>
<img src="./projects/Skin-Cancer-HAM10000/test/vasc___ISIC_0029448_101.jpg"  width="400" height="auto"><br><br>
<br>
<h3>
<a id="5.3">5.3 Inference result</a>
</h3>
This inference command will generate <a href="./projects/Skin-Cancer-HAM10000/inference/inference.csv">inference result file</a>.
<br>At this time, you can see the inference accuracy for the test dataset by our trained model is very low.
More experiments will be needed to improve accuracy.<br>

<br>
Inference console output:<br>
<img src="./asset/Skin_Cancer_infer_console_output_at_epoch_28_0813.png" width="740" height="auto"><br>
<br>

Inference result (inference.csv):<br>
<img src="./asset/Skin_Cancer_inference_at_epoch_28_0813.png" width="740" height="auto"><br>
<br>
<h2>
<a id="6">6 Evaluation</a>
</h2>
<h3>
<a id="6.1">6.1 Evaluation script</a>
</h3>
Please run the following bat file to evaluate <a href="./projects/Skin-Cancer/Resampled_HAM10000/Testing">
Resampled_HAM10000/Testing dataset</a> by the trained model.<br>
<pre>
./3_evaluate.bat
</pre>
<pre>
rem 3_evaluate.bat
python ../../EfficientNetV2Evaluator.py ^
  --model_name=efficientnetv2-m  ^
  --model_dir=./models ^
  --data_dir=./Resampled_HAM10000/Testing ^
  --evaluation_dir=./evaluation ^
  --fine_tuning=True ^
  --trainable_layers_ratio=0.4 ^
  --eval_image_size=480 ^
  --mixed_precision=True ^
  --debug=False 
</pre>
<br>

<h3>
<a id="6.2">6.2 Evaluation result</a>
</h3>

This evaluation command will generate <a href="./projects/Skin-Cancer/evaluation/classification_report.csv">a classification report</a>
 and <a href="./projects/Skin-Cancer-HAM10000/evaluation/confusion_matrix.png">a confusion_matrix</a>.
<br>
<br>
Evaluation console output:<br>
<img src="./asset/Skin_Cancer_evaluate_console_output_at_epoch_28_0813.png" width="740" height="auto"><br>
<br>

<br>
Classification report:<br>
<img src="./asset/Skin_Cancer_classification_report_at_epoch_28_0813.png" width="740" height="auto"><br>
<br>
Confusion matrix:<br>
<img src="./projects/Skin-Cancer-HAM10000/evaluation/confusion_matrix.png" width="740" height="auto"><br>

