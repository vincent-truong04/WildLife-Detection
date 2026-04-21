import { useEffect, useState } from 'react';
import { ref, listAll, getDownloadURL, getMetadata } from 'firebase/storage';
import { storage } from './firebase';
import './App.css';

function App() {
  const [images, setImages] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedImage, setSelectedImage] = useState(null);
  
  // We keep your clean, hardcoded dropdown options
  const folders = ['detected', 'empty'];
  const [currentFolder, setCurrentFolder] = useState(folders[0]);

  // --- Step 2. Fetch the images (and look inside subfolders!) ---
  useEffect(() => {
    if (!currentFolder) return;

    setLoading(true);
    const folderRef = ref(storage, `${currentFolder}/`);

    // --- NEW: Helper function to grab files from subfolders automatically ---
    const fetchAllImagesInFolder = async (dirRef) => {
      let allFiles = [];
      const response = await listAll(dirRef);

      // 1. Grab any loose images directly in this folder
      allFiles = [...allFiles, ...response.items];

      // 2. Loop through every individual subfolder and grab the images inside them too
      for (const subfolder of response.prefixes) {
        const subFiles = await fetchAllImagesInFolder(subfolder);
        allFiles = [...allFiles, ...subFiles];
      }

      return allFiles;
    };

    // Use our new helper function instead of standard listAll
    fetchAllImagesInFolder(folderRef)
      .then(async (allItemRefs) => {
        const promises = allItemRefs.map(async (item) => {
          const url = await getDownloadURL(item);
          const metadata = await getMetadata(item);
          return {
            url: url,
            name: item.name,
            created: metadata.timeCreated,
          };
        });

        const fileData = await Promise.all(promises);
        
        // Sort them chronologically so the newest detections are always first
        const sortedFiles = fileData.sort((a, b) => new Date(b.created) - new Date(a.created));

        setImages(sortedFiles);
        setLoading(false);
      })
      .catch((error) => {
        console.error("Error loading images:", error);
        setLoading(false);
      });
  }, [currentFolder]);

  return (
    <div className="container">
      <header>
        <h1>Wildlife Detection Feed</h1>
        
        <div className="folder-select">
          <label htmlFor="folders" style={{ fontWeight: 'bold', marginRight: '10px' }}>
            Select Folder: 
          </label>
          <select 
            id="folders" 
            value={currentFolder} 
            onChange={(e) => setCurrentFolder(e.target.value)}
            style={{ padding: '8px', borderRadius: '5px', fontSize: '16px', border: '1px solid #ccc' }}
          >
            {folders.map((folder, index) => (
              <option key={index} value={folder}>
                {folder.charAt(0).toUpperCase() + folder.slice(1)}
              </option>
            ))}
          </select>
        </div>
        
        <p>{images.length} detections found in <strong>{currentFolder}</strong></p>
      </header>

      {loading ? (
        <p className="loading">Loading feed...</p>
      ) : (
        <div className="grid">
          {images.map((img, index) => (
            <div key={index} className="card">
              <img 
                src={img.url} 
                alt="Wildlife Detection" 
                onClick={() => setSelectedImage(img.url)}
                style={{ cursor: 'pointer' }}
              />
              <div className="info">
                <p className="date">
                  {new Date(img.created).toLocaleString()}
                </p>
                <p className="filename">{img.name}</p>
              </div>
            </div>
          ))}
        </div>
      )}

      {selectedImage && (
        <div className="modal" onClick={() => setSelectedImage(null)}>
          <span className="close-btn">&times;</span>
          <img className="modal-content" src={selectedImage} alt="Enlarged Wildlife" />
        </div>
      )}
    </div>
  );
}

export default App;