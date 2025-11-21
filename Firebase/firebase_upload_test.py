import firebase_admin
from firebase_admin import credentials, storage

# 1️⃣ Point to your downloaded JSON key file
cred = credentials.Certificate("/home/raspberrypi/yolo_project/WildLife-Detection/Firebase/real-time-wildlife-detector-firebase-adminsdk-fbsvc-7c1cbed963.json")
firebase_admin.initialize_app(cred, {
    "storageBucket": "real-time-wildlife-detector.firebasestorage.app"
})

# 2️⃣ Choose an image on your computer to upload
file_path = "/home/raspberrypi/Downloads/DeerUploadTest.jpg"  # <-- put the image name here
firebase_path = "deer/DeerUploadTest.jpg" #uploads to deer folder in firebase

bucket = storage.bucket()
blob = bucket.blob(firebase_path)  # path inside Firebase Storage

# 3️⃣ Upload the image
blob.upload_from_filename(file_path)
print("firebase upload successful")
