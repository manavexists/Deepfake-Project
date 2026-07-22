import os
import io
import cv2
import base64
import tempfile
from PIL import Image, ImageChops, ImageEnhance, ExifTags
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline

app = FastAPI(title="Local Deepfake Detection Engine")

# Enable CORS so your frontend UI can communicate with localhost:8000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading local AI Vision Model into memory...")
ai_detector = pipeline("image-classification", model="dima806/deepfake_vs_real_image_detection")
print("✅ Local AI Model Ready!")

# --- PIPELINE BRANCH 1: METADATA ANALYSIS ---
def analyze_metadata(pil_image):
    meta_dict = {}
    score = 0.1 # Baseline low risk
    try:
        exif = pil_image._getexif()
        if exif:
            for tag_id, value in exif.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                meta_dict[str(tag)] = str(value)[:60]
                if tag == "Software" and any(sus in str(value).lower() for sus in ["photoshop", "gimp", "lightroom", "midjourney", "dall-e"]):
                    score = 0.85
        else:
            meta_dict["EXIF"] = "No EXIF camera metadata found (Likely stripped or synthetic)."
            score = 0.50
    except Exception:
        meta_dict["Error"] = "Failed to parse EXIF metadata."
        score = 0.50
        
    return score, meta_dict

# --- PIPELINE BRANCH 2: AI DETECTION MODEL ---
def run_ai_detection(pil_image):
    results = ai_detector(pil_image)
    fake_score = next((res['score'] for res in results if 'fake' in res['label'].lower()), 0.1)
    return fake_score

# --- PIPELINE BRANCH 3: DIGITAL FORENSICS (ELA) ---
def run_digital_forensics(pil_image):
    img_rgb = pil_image.convert('RGB')
    
    # Save a temporary JPEG in memory at 90% quality
    buffer = io.BytesIO()
    img_rgb.save(buffer, format='JPEG', quality=90)
    buffer.seek(0)
    compressed = Image.open(buffer)
    
    # Calculate pixel level difference
    ela_image = ImageChops.difference(img_rgb, compressed)
    
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0: max_diff = 1
    
    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    forensic_score = min(0.95, max_diff / 255.0)
    
    # Convert ELA image to Base64 so your UI can display the heatmap directly
    ela_buffer = io.BytesIO()
    ela_image.save(ela_buffer, format="JPEG")
    ela_b64 = base64.b64encode(ela_buffer.getvalue()).decode('utf-8')
    
    return forensic_score, f"data:image/jpeg;base64,{ela_b64}"

# --- EVIDENCE FUSION ENGINE ---
def evidence_fusion(meta_score, ai_score, forensic_score):
    return (meta_score * 0.15) + (ai_score * 0.55) + (forensic_score * 0.30)

# --- VIDEO PROCESSOR ---
def process_video_frames(video_path, num_frames=5):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        raise ValueError("Cannot read video file.")
        
    interval = max(1, total_frames // num_frames)
    ai_scores, forensic_scores = [], []
    last_ela_b64 = None

    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            ai_scores.append(run_ai_detection(pil_img))
            f_score, ela_b64 = run_digital_forensics(pil_img)
            forensic_scores.append(f_score)
            last_ela_b64 = ela_b64

    cap.release()
    
    avg_ai = sum(ai_scores) / len(ai_scores) if ai_scores else 0.1
    avg_forensic = sum(forensic_scores) / len(forensic_scores) if forensic_scores else 0.1
    
    return avg_ai, avg_forensic, last_ela_b64

# --- API ENDPOINT ---
@app.post("/analyze")
async def analyze_media(file: UploadFile = File(...)):
    file_ext = os.path.splitext(file.filename)[1]
    mime_type = file.content_type or ""

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
        temp_file.write(await file.read())
        temp_path = temp_file.name

    try:
        # IMAGE PROCESSING
        if mime_type.startswith("image/"):
            raw_image = Image.open(temp_path)
            rgb_image = raw_image.convert("RGB")
            
            meta_score, meta_details = analyze_metadata(raw_image)
            ai_score = run_ai_detection(rgb_image)
            forensic_score, ela_b64 = run_digital_forensics(rgb_image)
            
            media_type = "image"

        # VIDEO PROCESSING
        elif mime_type.startswith("video/"):
            meta_score, meta_details = 0.5, {"Info": "Video metadata stripped during frame analysis"}
            ai_score, forensic_score, ela_b64 = process_video_frames(temp_path)
            media_type = "video"

        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Upload JPG, PNG, or MP4/MOV.")

        # Evidence Fusion Calculation
        final_score = evidence_fusion(meta_score, ai_score, forensic_score)
        
        verdict = "Fake" if final_score > 0.65 else ("Suspicious" if final_score > 0.4 else "Authentic")

        # Payload sent directly to your custom frontend
        return {
            "status": "success",
            "media_type": media_type,
            "confidence_score": round(final_score, 3),
            "verdict": verdict,
            "xai_report": {
                "ai_model": {
                    "score": round(ai_score, 3),
                    "description": "Spatial artifact evaluation via Vision Transformer."
                },
                "forensics": {
                    "score": round(forensic_score, 3),
                    "ela_heatmap_base64": ela_b64,
                    "description": "Compression discrepancy analysis (Error Level Analysis)."
                },
                "metadata": {
                    "score": round(meta_score, 3),
                    "details": meta_details
                }
            }
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
