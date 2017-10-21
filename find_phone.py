import keras
import numpy as np
import sys
from skimage.io import imread
from keras.models import model_from_json
import keras.backend as K
import os
def proximity_loss(y_true, y_pred):
    y_true = K.l2_normalize(y_true)
    print(type(y_pred))
    y_pred = K.l2_normalize(y_pred)
    return K.mean(K.sqrt(2. - (2 * K.sum((y_true * y_pred), axis=-1))))


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("weights.h5")
# evaluate loaded model on test data
#loaded_model.compile(loss='mse', optimizer=keras.optimizers.SGD(lr= 0.0001))

test_img = sys.argv[1]
filename = test_img.split('/')[-1]

img =np.array( imread(test_img), dtype=np.float32)
lab = test_img.split('/')[:-1]
lab.append('labels.txt')
lab_file = os.path.join(*lab)
with open(lab_file, 'r') as fp:
    for lines in fp.readlines():
        dat = lines.split(' ')
        if(dat[0] == filename):
            _, x, y = dat
            break
target = np.asarray([x, y], dtype=np.float32)

img = keras.utils.normalize(img)
pred = loaded_model.predict(img.reshape(1, 326, 490, 3), batch_size=1)
result = np.linalg.norm(np.subtract(pred, target), ord ='fro') * 0.25

print("Test result: ", result)
if result <= 0.05:
    print("Phone Detected", result)
else:
    print("Phone Not Detected", result)



