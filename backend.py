import os
import tensorflow as tf
import numpy as np
import PIL.Image
import tensorflow_hub as hub
from flask.templating import render_template
from flask import Flask, request


os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)

def load_img(path_to_img):
  max_dim = 512
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img



app = Flask(__name__)
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

@app.route('/',methods =["GET", "POST"])
def index():
  if request.method == 'POST':
    image_url = request.form.get('input_Image')
    Styleimage_url = request.form.get('style_Image')
    content_image = load_img(request.form.get('input_Image'))
    style_image = load_img(request.form.get('style_Image'))
    stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
    output_image = tensor_to_image(stylized_image)
    output_file_name = 'output.png'
    output_image.save(f'./static/{output_file_name}')
    return render_template('index.html',value=output_file_name,input=image_url,style=Styleimage_url)
  return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)
    