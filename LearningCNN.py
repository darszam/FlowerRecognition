from CreatingCNN import *

print(os.listdir("flowers_to_rec/"))
print(os.listdir("flowers/"))

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

training_set = train_datagen.flow_from_directory('flowers/',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

flower_recognizer_model.fit_generator(training_set, steps_per_epoch=100, epochs=10)

flower_recognizer_model.save('flower_recognition_model.h5')
