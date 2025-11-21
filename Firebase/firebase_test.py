import firebase_admin
from firebase_admin import credentials, storage

print("Starting firebase test")

cred = credentials.Certificate("/home/raspberrypi/yolo_project/WildLife-Detection/Firebase/real-time-wildlife-detector-firebase-adminsdk-fbsvc-7c1cbed963.json")
firebase_admin.initialize_app(cred, {"storageBucket": "real-time-wildlife-detector.firebasestorage.app"})
bucket = storage.bucket()
print("Connected to Firebase Storage")
