from ultralytics import YOLO
model = YOLO("attack object detection/OD-WeaponDetection-master/Weapons and similar handled objects/best.pt")
model.predict(source="https://cdn-dnjgd.nitrocdn.com/gMwOVjQTBBAUgeYRRMkiUpfNoASsHPBd/assets/images/optimized/rev-6d393b2/noblie.eu/wp-content/uploads/2020/12/Best-Hunting-Knives.jpeg",save=True)