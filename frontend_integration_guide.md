# 🎨 Frontend Integration Guide: Deepfake & Misinformation API

This guide provides the API endpoints, payload contracts, response formats, and React examples for integrating the frontend with the FastAPI backend.

---

## 🚀 1. Endpoint: Analyze Media (`POST /analyze`)
Processes images, audio, and videos for deepfakes, generates an ELA heatmap, and calculates the Authenticity Trust Score.

* **Method**: `POST`
* **Content-Type**: `multipart/form-data`
* **Request Body**:
  * `file`: `UploadFile` (Required) — Supported formats: JPG, PNG, WebP, MP3, WAV, MP4, AVI, MOV.

### 📥 Response Payload
```json
{
  "report_id": "D3A1B4F8",
  "media_type": "image",
  "trust_score": 82,
  "verdict": "Highly Authentic",
  "manipulation_types": ["None Detected"],
  "download_report_url": "/download-report/D3A1B4F8",
  "scores": {
    "ai_probability": 12.4,
    "metadata_risk": 10.0,
    "forensic_risk": 15.0
  },
  "explainable_ai": {
    "heatmap_base64": "data:image/jpeg;base64,...",
    "anomalies": {},
    "face_detection": {
      "face_count": 1,
      "faces": [
        {
          "bbox": [120, 80, 240, 240], 
          "eyes_detected": 2
        }
      ],
      "anomaly_score": 0.0,
      "message": "Detected 1 face(s)."
    }
  }
}
```

### ⚛️ React Bounding Box & Heatmap Integration Example
When displaying face bounding boxes, you must scale the coordinates `[x, y, w, h]` relative to the rendered image. Use a container with `position: relative` and place absolute `div` elements on top of the image using percentages or container scale mapping.

```jsx
import React, { useRef, useState, useEffect } from 'react';
import axios from 'axios';

export function MediaAnalyzer() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const imgRef = useRef(null);
  const [scale, setScale] = useState({ x: 1, y: 1 });

  // Recalculates coordinates if image is scaled responsively (e.g. max-width: 100%)
  const handleImageLoad = () => {
    if (imgRef.current) {
      const { naturalWidth, naturalHeight, clientWidth, clientHeight } = imgRef.current;
      setScale({
        x: clientWidth / naturalWidth,
        y: clientHeight / naturalHeight
      });
    }
  };

  const handleUpload = async () => {
    const formData = new FormData();
    formData.append('file', file);
    const res = await axios.post('http://localhost:8000/analyze', formData);
    setResult(res.data);
  };

  return (
    <div className="glass-card p-6 rounded-2xl max-w-4xl mx-auto">
      <input type="file" onChange={(e) => setFile(e.target.files[0])} />
      <button onClick={handleUpload} className="bg-teal-500 hover:bg-teal-600 text-white px-4 py-2 rounded-lg mt-4">
        Run Analysis
      </button>

      {result && (
        <div className="mt-8 grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Left: Interactive Image Preview with Bounding Boxes */}
          <div className="relative inline-block overflow-hidden rounded-xl border border-gray-700" style={{ position: 'relative' }}>
            <img 
              ref={imgRef} 
              src={URL.createObjectURL(file)} 
              alt="Source" 
              onLoad={handleImageLoad}
              className="w-full h-auto"
            />
            {/* Draw Bounding Boxes */}
            {result.explainable_ai.face_detection?.faces.map((face, i) => {
              const [x, y, w, h] = face.bbox;
              const hasAnomaly = face.eyes_detected !== 2;
              return (
                <div 
                  key={i}
                  style={{
                    position: 'absolute',
                    left: `${x * scale.x}px`,
                    top: `${y * scale.y}px`,
                    width: `${w * scale.x}px`,
                    height: `${h * scale.y}px`,
                    border: hasAnomaly ? '2px solid #ef4444' : '2px dashed #10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.05)',
                    boxShadow: hasAnomaly ? '0 0 10px rgba(239, 68, 68, 0.5)' : '0 0 10px rgba(16, 185, 129, 0.3)',
                    transition: 'all 0.3s ease-in-out'
                  }}
                >
                  <span className={`absolute top-0 left-0 text-[10px] px-1 font-bold text-white ${hasAnomaly ? 'bg-red-500' : 'bg-emerald-500'}`}>
                    Face {i + 1} {hasAnomaly && "(Anomaly)"}
                  </span>
                </div>
              );
            })}
          </div>

          {/* Right: ELA Heatmap Display */}
          {result.explainable_ai.heatmap_base64 && (
            <div className="flex flex-col gap-2">
              <h3 className="text-gray-300 font-semibold">Forensic Heatmap (ELA)</h3>
              <img 
                src={result.explainable_ai.heatmap_base64} 
                alt="ELA Heatmap" 
                className="w-full h-auto rounded-xl border border-gray-700"
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
}
```

---

## 🔎 2. Endpoint: Fact-Checking Assistant (`POST /fact-check`)
Searches trusted news and provides context checking.

* **Method**: `POST`
* **Content-Type**: `application/x-www-form-urlencoded` or `multipart/form-data`
* **Request Body**:
  * `claim`: `string` (Required) — The text statement or caption to fact-check.
  * `file`: `UploadFile` (Optional) — Associated image or context document.

### 📥 Response Payload
```json
{
  "status": "Misleading Claim/Media Detected",
  "original_claim": "Delhi flooded today",
  "reverse_image_search": {
    "match_found": true,
    "first_website": "reuters.com",
    "top_sources": [
      "Debunked: Old 2021 flooding video shared as recent Delhi rains",
      "Fact Check: Delhi waterlogging claim uses old Bangladesh clip"
    ],
    "source_links": [
      "https://reuters.com/article/factcheck-delhi-floods",
      "https://bbc.com/news/world-asia-58213812"
    ]
  },
  "conclusion": "Found matching reports online from sources like reuters.com. Top article: 'Debunked: Old 2021 flooding video shared as recent Delhi rains'. Context details: Video has been shared since 2021 and was recorded in Bangladesh."
}
```

### ⚛️ React Fact Check UI Example
```jsx
export function FactChecker() {
  const [claim, setClaim] = useState('');
  const [response, setResponse] = useState(null);

  const checkClaim = async () => {
    const formData = new FormData();
    formData.append('claim', claim);
    const res = await axios.post('http://localhost:8000/fact-check', formData);
    setResponse(res.data);
  };

  return (
    <div className="p-6 bg-slate-900 rounded-xl text-white">
      <input 
        type="text" 
        value={claim} 
        onChange={(e) => setClaim(e.target.value)} 
        placeholder="Enter rumor or claim..."
        className="w-full p-3 bg-slate-800 rounded border border-gray-700"
      />
      <button onClick={checkClaim} className="mt-4 px-6 py-2 bg-blue-600 rounded hover:bg-blue-700">
        Analyze Claim
      </button>

      {response && (
        <div className="mt-6 border-t border-gray-700 pt-4">
          <div className={`text-lg font-bold ${response.status.includes('Misleading') ? 'text-red-400' : 'text-green-400'}`}>
            {response.status}
          </div>
          <p className="mt-2 text-gray-300 italic">"{response.conclusion}"</p>
          
          <div className="mt-4">
            <h4 className="font-semibold text-gray-400">Sources Cited:</h4>
            <ul className="list-disc pl-5 mt-2">
              {response.reverse_image_search.top_sources.map((src, i) => (
                <li key={i}>
                  <a href={response.reverse_image_search.source_links[i]} target="_blank" rel="noopener noreferrer" className="text-teal-400 hover:underline">
                    {src}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}
```

---

## 📥 3. Endpoint: Download PDF Report (`GET /download-report/{report_id}`)
Downloads the official generated PDF report.

* **Method**: `GET`
* **Response**: Returns a PDF File Response.
* **React Integration**: Point a button link directly to this URL to trigger browser download:
  ```jsx
  <a 
    href={`http://localhost:8000/download-report/${result.report_id}`} 
    download
    className="bg-red-500 hover:bg-red-600 text-white font-bold py-2 px-4 rounded-xl flex items-center gap-2"
  >
    📥 Download Verification PDF
  </a>
  ```

---

## 📈 4. Endpoint: Analytics Dashboard (`GET /analytics`)
Provides data to build live dashboards, showing total checked, fakes detected, and daily uploads.

* **Method**: `GET`
* **Response Payload**:
  ```json
  {
    "total_checked": 1245,
    "fake_detected": 893,
    "real_verified": 352,
    "common_manipulations": ["Face Swap", "Voice Cloning", "Compression Artifacts"],
    "daily_activity": [120, 150, 95, 205, 310, 180, 183]
  }
  ```
* **React Charting Tip**: Pass the array `daily_activity` directly into charts (like Recharts or Chart.js) to show a live activity trendline!

---

## 🎨 5. Recommended UI Themes & Visual Cues
To win a hackathon, style the app using **sleek developer trends**:
* **Theme**: Sleek Dark Mode (e.g. background `#0d0f17`, cards `#161b26`).
* **Visual Trust Indicator**: Use a circular SVG stroke animation for the **Authenticity Trust Score** (0-100).
* **Alert Statuses**:
  * **90-100**: Green (`#10b981`) — *"Highly Authentic"*
  * **50-80**: Yellow/Orange (`#f59e0b`) — *"Suspicious Context"*
  * **0-49**: Red (`#ef4444`) — *"Confirmed Manipulation"*
* **ELA Display**: Position ELA Heatmap and Original image side-by-side or use a slider element to let users wipe between them to compare pixels.
