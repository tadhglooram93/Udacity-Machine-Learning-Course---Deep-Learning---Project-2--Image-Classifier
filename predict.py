from PIL import Image 
import numpy as np 
import pandas as pd
import argparse 
import tensorflow as tf
import tensorflow_hub as hub
import warnings
warnings.filterwarnings('ignore')
import logging 
import json

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

#set the argument parameters
parser = argparse.ArgumentParser ()
parser.add_argument ('--image_path', default='./test_images/hard-leaved_pocket_orchid.jpg', help = 'Path to image file you want to make a prediction of', type = str)
parser.add_argument('--model', default='./immage_classification_flower_model.h5',help='Path of your model', type=str)
parser.add_argument ('--top_k', default = 5, help = 'Top K most likely classes of probability for selected input', type = int)
commands = parser.parse_args()

#loads the json file with the names of the flowers and their Ids
with open('label_map.json', 'r') as f:
    class_names = json.load(f)

#Create the function for processing the images
def process_image(image):
    img = tf.convert_to_tensor(image, dtype=tf.float32)
    img = tf.image.resize(img, (224, 224))
    img /= 255
    return img.numpy()

#create the function that will load the image, process it,load our keras model, predict and than create a dataframe consisting of the top_k
def predict(image_path, model, top_k):
    img = Image.open(image_path)
    img = np.asarray(img)
    img = process_image(img)
    model =  tf.keras.models.load_model(model, custom_objects={'KerasLayer':hub.KerasLayer})
    pred = model.predict(np.expand_dims(img,axis=0))
    pred_df = pd.DataFrame(pred[0]).reset_index()
    pred_df = pred_df.rename(columns={'index':'Class_id',0:'Probability'})
    pred_df['Class_id'] = pred_df['Class_id']+1
    pred_df['Class_id'] = pred_df['Class_id'].apply(str)
    pred_df['Class_name'] = pred_df['Class_id'].map(class_names)
    pred_df = pred_df.sort_values(by='Probability', ascending=False).head(top_k)
    actual_flower = 'Actual Flower: '+ image_path.split('/')[2].replace('.jpg','')
    return pred_df, actual_flower
                                                                        
if __name__ == "__main__":
    print(predict(image_path, model, top_k))                                                                                                             