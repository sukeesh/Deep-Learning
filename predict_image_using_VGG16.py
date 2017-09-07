import numpy as np
from PIL import Image

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions

model = VGG16(weights='imagenet', include_top=True)
layers = dict([(layer.name, layer.output) for layer in model.layers])

image_path = 'images/peacock.jpg'
image = Image.open(image_path)
image = image.resize((224, 224))

x = np.asarray(image, dtype='float32')
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
preds = model.predict(x)

print '\n'
print('Predicted:', decode_predictions(preds, top=3)[0])
