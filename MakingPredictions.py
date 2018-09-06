from CreatingCNN import *

flower_recognizer_model.load_weights('flower_recognition_model.h5')

image_to_recognize = image.load_img('flowers_to_rec/purple_rose.jpg', target_size=(64,64))
image_to_recognize = image.img_to_array(image_to_recognize)
image_to_recognize = np.expand_dims(image_to_recognize, axis = 0)
prediction = flower_recognizer_model.predict(image_to_recognize)

print(os.listdir("flowers/"))
print(prediction)