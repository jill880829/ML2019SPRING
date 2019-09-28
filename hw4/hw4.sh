wget https://github.com/b06901087/Models/releases/download/HW3-model/keras_model1.h5
python hw4_saliency_map.py $1 $2
python hw4_visualize.py $1 $2
python hw4_lime.py $1 $2