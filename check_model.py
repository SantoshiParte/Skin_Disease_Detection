from ultralytics import YOLO

model = YOLO("models/best.pt")

print("Model task:", model.task)
print("\nTotal classes:", len(model.names))

print("\nAvailable disease classes:")

for class_id, class_name in model.names.items():
    print(class_id, ":", class_name)