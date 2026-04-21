
import { initializeApp } from "firebase/app";
import { getStorage } from "firebase/storage"; // <--- We added this line

const firebaseConfig = {
  apiKey: "AIzaSyBZA6ZGRybFbd4LDdOF6YW_O1e5w7RLjk8",
  authDomain: "real-time-wildlife-detector.firebaseapp.com",
  projectId: "real-time-wildlife-detector",
  storageBucket: "real-time-wildlife-detector.firebasestorage.app",
  messagingSenderId: "396594451829",
  appId: "1:396594451829:web:103e1fc5ad56e654b5ef3c",
  measurementId: "G-ZHER7TW4W2"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);

// Initialize Cloud Storage and export it so App.jsx can use it
export const storage = getStorage(app);