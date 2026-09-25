import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model("crop_disease_model.h5")

class_names = [
    "Potato Common Scab",
    "Potato Early Blight",
    "Potato Healthy",
    "Potato Late Blight",
    "Tomato Healthy",
    "Tomato Late Blight",
    "Tomato Leaf Mold",
    "Tomato Early Blight"
]

img_path = "test.jpg"


img = image.load_img(img_path, target_size=(128,128))

img_array = image.img_to_array(img)


img_array = np.expand_dims(img_array, axis=0)


img_array = img_array / 255.0

prediction = model.predict(img_array)

predicted_class = np.argmax(prediction)

print("Prediction:", class_names[predicted_class])

confidence = np.max(prediction) * 100
print(f"Confidence: {confidence:.2f}%")