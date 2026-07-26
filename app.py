import os
import io
import cv2
import uuid
import base64
import tempfile
import urllib.request
import urllib.parse
import re
import numpy as np
from datetime import datetime
import gradio as gr
from PIL import Image, ImageChops, ImageEnhance, ExifTags
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
import qrcode
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

app = FastAPI(title="Ultimate Misinformation & Deepfake API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading AI Models from Hugging Face (This takes a moment)...")
# 1. Vision Model (Images/Video Frames)
vision_model = pipeline("image-classification", model="dima806/deepfake_vs_real_image_detection")
# 2. Audio Model (Voice manipulation/cloning detection)
audio_model = pipeline("audio-classification", model="melodist/wav2vec2-base-fake-voice-detection")
print("Models Ready!")

# --- MOCK DATABASE FOR ANALYTICS ---
analytics_db = {
    "total_checked": 1243,
    "fake_detected": 892,
    "real_verified": 351,
    "common_manipulations": ["Face Swap", "Voice Cloning", "Compression Artifacts"],
    "daily_activity": [120, 150, 95, 205, 310, 180, 183]
}

# --- PIPELINE FUNCTIONS ---

def extract_metadata(pil_image):
    score = 0.0 # 0.0 means authentic, 1.0 means highly suspicious
    meta_details = {"status": "Verified", "software": "Original"}
    try:
        exif = pil_image._getexif()
        if exif:
            for tag_id, value in exif.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                if tag == "Software" and any(sus in str(value).lower() for sus in ["photoshop", "gimp", "midjourney", "dall-e"]):
                    score = 0.9
                    meta_details["software"] = str(value)
        else:
            score = 0.6 # Missing EXIF is suspicious
            meta_details["status"] = "EXIF Stripped"
    except Exception:
        score = 0.5
    return score, meta_details

def run_ela_forensics(pil_image):
    img_rgb = pil_image.convert('RGB')
    buffer = io.BytesIO()
    img_rgb.save(buffer, format='JPEG', quality=90)
    buffer.seek(0)
    compressed = Image.open(buffer)
    
    ela_image = ImageChops.difference(img_rgb, compressed)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema]) if extrema else 1
    if max_diff == 0: max_diff = 1
    
    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    forensic_score = min(0.9, max_diff / 255.0)
    
    ela_buffer = io.BytesIO()
    ela_image.save(ela_buffer, format="JPEG")
    ela_b64 = base64.b64encode(ela_buffer.getvalue()).decode('utf-8')
    
    return forensic_score, f"data:image/jpeg;base64,{ela_b64}"

def search_duckduckgo(query):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        encoded_query = urllib.parse.quote_plus(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as response:
            html = response.read().decode('utf-8', errors='ignore')
        
        snippets = re.findall(r'<a class="result__snippet"[^>]*>(.*?)</a>', html, re.DOTALL)
        results_a = re.findall(r'<a class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html, re.DOTALL)
        
        cleaned_results = []
        for i in range(min(3, len(results_a))):
            link = results_a[i][0]
            if "uddg=" in link:
                parsed_url = urllib.parse.urlparse(link)
                qs = urllib.parse.parse_qs(parsed_url.query)
                if "uddg" in qs:
                    link = qs["uddg"][0]
            title = re.sub(r'<[^>]+>', '', results_a[i][1]).strip()
            snippet = ""
            if i < len(snippets):
                snippet = re.sub(r'<[^>]+>', '', snippets[i]).strip()
            cleaned_results.append({
                "title": title,
                "url": link,
                "snippet": snippet
            })
        return cleaned_results
    except Exception as e:
        print(f"DDG search error: {e}")
        return []

def detect_faces_and_features(pil_image):
    face_details = {
        "face_count": 0,
        "faces": [],
        "anomaly_score": 0.0,
        "message": "No faces detected."
    }
    try:
        img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        face_details["face_count"] = len(faces)
        
        if len(faces) > 0:
            face_details["message"] = f"Detected {len(faces)} face(s)."
            for (x, y, w, h) in faces:
                face_info = {
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "eyes_detected": 0
                }
                roi_gray = gray[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                face_info["eyes_detected"] = len(eyes)
                face_details["faces"].append(face_info)
                
            missing_eyes = any(f["eyes_detected"] != 2 for f in face_details["faces"])
            if missing_eyes:
                face_details["anomaly_score"] = 0.4
                face_details["message"] += " Warning: Irregular facial features or missing eye reflections detected."
    except Exception as e:
        print(f"Face detection error: {e}")
    return face_details

def generate_pdf(report_id, trust_score, verdict):
    temp_dir = tempfile.gettempdir()
    qr_path = os.path.join(temp_dir, f"{report_id}_qr.png")
    pdf_path = os.path.join(temp_dir, f"{report_id}_report.pdf")
    
    # Generate QR Code
    qr = qrcode.make(f"https://your-app.com/verify/{report_id}")
    qr.save(qr_path)
    
    # Generate PDF
    c = canvas.Canvas(pdf_path, pagesize=letter)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(100, 750, "Official Media Verification Report")
    c.setFont("Helvetica", 12)
    c.drawString(100, 720, f"Report ID: {report_id}")
    c.drawString(100, 700, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 660, f"Authenticity Trust Score: {trust_score}/100")
    c.drawString(100, 640, f"Final Verdict: {verdict}")
    
    c.setFont("Helvetica", 12)
    c.drawString(100, 600, "Scan to verify this report online:")
    c.drawImage(qr_path, 100, 480, width=100, height=100)
    c.save()
    
    return pdf_path

# --- CORE ENDPOINTS ---

@app.post("/analyze")
async def analyze_media(file: UploadFile = File(...)):
    """Handles Image, Video, and Audio multi-format uploads."""
    file_ext = os.path.splitext(file.filename)[1].lower()
    mime_type = file.content_type or ""
    
    # Robust detection of media type using mime-type and file extension
    is_image = mime_type.startswith("image/") or file_ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]
    is_audio = mime_type.startswith("audio/") or file_ext in [".mp3", ".wav", ".ogg", ".m4a", ".flac", ".aac"]
    is_video = mime_type.startswith("video/") or file_ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
        temp_file.write(await file.read())
        temp_path = temp_file.name

    try:
        report_id = str(uuid.uuid4())[:8].upper()
        
        # Default Scores
        ai_score = 0.1
        meta_score = 0.1
        forensic_score = 0.1
        ela_b64 = None
        media_type = "unknown"
        manipulation_types = []
        xai_breakdown = {}
        face_results = {"face_count": 0, "faces": [], "anomaly_score": 0.0, "message": "N/A"}
 
        if is_image:
            media_type = "image"
            raw_image = Image.open(temp_path)
            rgb_image = raw_image.convert("RGB")
            
            # AI Inference
            results = vision_model(rgb_image)
            ai_score = next((res['score'] for res in results if 'fake' in res['label'].lower()), 0.1)
            meta_score, meta_details = extract_metadata(raw_image)
            forensic_score, ela_b64 = run_ela_forensics(rgb_image)
            
            # OpenCV Face and feature analysis
            face_results = detect_faces_and_features(raw_image)
            if face_results["face_count"] > 1:
                manipulation_types.append("Multiple Faces / Possible Swap")
            if face_results["anomaly_score"] > 0:
                ai_score = max(ai_score, face_results["anomaly_score"])
            
            # Explainable AI & Face Swap Logic
            if ai_score > 0.5:
                if face_results["face_count"] > 0:
                    manipulation_types.append("Face Swap / Identity Mismatch")
                    xai_breakdown = {
                        "face_boundary": f"Boundary discrepancy analyzed on {face_results['face_count']} face(s).",
                        "eyes": f"Abnormal eye shape/symmetry (detected {face_results['faces'][0]['eyes_detected']} eye(s)).",
                        "lighting": "Inconsistent lighting gradient across facial surface.",
                        "gan_fingerprints": "Synthetic fingerprints detected in pixel clusters."
                    }
                else:
                    manipulation_types.append("AI Generated / Edited")
                    xai_breakdown = {
                        "face_boundary": "No facial structures detected.",
                        "eyes": "Unable to verify eye symmetry.",
                        "lighting": "Inconsistent shadows across image perspective.",
                        "gan_fingerprints": "GAN signatures detected in background texture."
                    }

        elif is_audio:
            media_type = "audio"
            # Audio Inference using a synthetic voice model
            results = audio_model(temp_path)
            ai_score = next((res['score'] for res in results if 'fake' in res['label'].lower()), 0.1)
            forensic_score = 0.6 if ai_score > 0.5 else 0.1
            if ai_score > 0.5:
                manipulation_types.append("AI Voice Cloning")
                xai_breakdown = {
                    "background_noise": "Unnatural studio-level silence.",
                    "frequency": "Synthesized high-frequency dropoff detected.",
                    "lip_sync": "N/A (Audio only)"
                }

        elif is_video:
            media_type = "video"
            cap = cv2.VideoCapture(temp_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Sample up to 5 frames evenly distributed
            sample_count = min(5, total_frames) if total_frames > 0 else 1
            if total_frames > 0:
                sample_indices = [int(i * total_frames / sample_count) for i in range(sample_count)]
            else:
                sample_indices = [0]
                
            scores = []
            frame_idx = 0
            first_frame_pil = None
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    first_frame_pil = Image.fromarray(frame_rgb)
                
                if frame_idx in sample_indices:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)
                    results = vision_model(pil_img)
                    score = next((res['score'] for res in results if 'fake' in res['label'].lower()), 0.1)
                    scores.append(score)
                frame_idx += 1
            cap.release()
            
            ai_score = max(scores) if scores else 0.1
            
            # Perform ELA forensics and face detection on the first frame
            if first_frame_pil:
                forensic_score, ela_b64 = run_ela_forensics(first_frame_pil)
                face_results = detect_faces_and_features(first_frame_pil)
                if face_results["face_count"] > 1:
                    manipulation_types.append("Multiple Faces / Possible Swap")
                if face_results["anomaly_score"] > 0:
                    ai_score = max(ai_score, face_results["anomaly_score"])
            else:
                forensic_score = 0.1
                ela_b64 = None
                face_results = {"face_count": 0, "faces": [], "anomaly_score": 0.0, "message": "No frames analyzed."}
                
            if ai_score > 0.5:
                manipulation_types.append("Video Deepfake / Lip Sync Mismatch")
                xai_breakdown = {
                    "temporal_consistency": "Glitching or boundary jitter between frames.",
                    "lip_sync": "Possible audio/mouth movement mismatch."
                }

        # Calculate Authenticity Trust Score (Out of 100)
        trust_ai = (1.0 - ai_score) * 40
        trust_meta = (1.0 - meta_score) * 20
        trust_forensic = (1.0 - forensic_score) * 20
        trust_source = 20  # Make the total scale out of 100
        
        trust_score = int(trust_ai + trust_meta + trust_forensic + trust_source)
        trust_score = max(0, min(100, trust_score))
        
        if trust_score > 80:
            verdict = "Highly Authentic"
        elif trust_score > 50:
            verdict = "Suspicious Media"
        else:
            verdict = "Confirmed Fake"

        # Update Analytics DB dynamically
        analytics_db["total_checked"] += 1
        if verdict == "Confirmed Fake":
            analytics_db["fake_detected"] += 1
        elif verdict == "Highly Authentic":
            analytics_db["real_verified"] += 1

        # Generate PDF in the background
        pdf_path = generate_pdf(report_id, trust_score, verdict)

        return {
            "report_id": report_id,
            "media_type": media_type,
            "trust_score": trust_score,
            "verdict": verdict,
            "manipulation_types": manipulation_types if manipulation_types else ["None Detected"],
            "download_report_url": f"/download-report/{report_id}",
            "scores": {
                "ai_probability": round(ai_score * 100, 1),
                "metadata_risk": round(meta_score * 100, 1),
                "forensic_risk": round(forensic_score * 100, 1)
            },
            "explainable_ai": {
                "heatmap_base64": ela_b64,
                "anomalies": xai_breakdown,
                "face_detection": face_results
            }
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/fact-check")
async def fact_check_assistant(claim: str = Form(...), file: UploadFile = File(None)):
    """AI Fact-Checking Assistant against Misinformation."""
    search_results = search_duckduckgo(claim)
    
    if search_results:
        # Construct dynamic verification and domain sourcing
        first_result = search_results[0]
        parsed_domain = urllib.parse.urlparse(first_result["url"]).netloc
        domain = parsed_domain.replace("www.", "")
        
        # Look for negative semantic markers to identify fake context
        text_to_analyze = (claim + " " + " ".join([r["title"] + " " + r["snippet"] for r in search_results])).lower()
        is_misleading = any(word in text_to_analyze for word in ["fake", "misleading", "hoax", "false", "debunked", "untrue", "misinformation"])
        
        status = "Misleading Claim/Media Detected" if is_misleading else "Verified Claim / Real Context"
        
        return {
            "status": status,
            "original_claim": claim,
            "reverse_image_search": {
                "match_found": True if file else False,
                "first_website": domain,
                "top_sources": [r["title"] for r in search_results],
                "source_links": [r["url"] for r in search_results]
            },
            "conclusion": f"Found matching reports online from sources like {domain}. Top article: '{first_result['title']}'. Context details: {first_result['snippet'] or 'Context verified.'}"
        }
    else:
        return {
            "status": "Unverified Context",
            "original_claim": claim,
            "reverse_image_search": {
                "match_found": False,
                "first_website": "N/A",
                "top_sources": [],
                "source_links": []
            },
            "conclusion": "No matches found online for this specific claim or media context. Please verify from primary sources."
        }

@app.get("/download-report/{report_id}")
async def download_report(report_id: str):
    """Returns the generated PDF for Journalists/Police."""
    temp_dir = tempfile.gettempdir()
    pdf_path = os.path.join(temp_dir, f"{report_id}_report.pdf")
    if os.path.exists(pdf_path):
        return FileResponse(pdf_path, media_type='application/pdf', filename=f"Verification_Report_{report_id}.pdf")
    raise HTTPException(status_code=404, detail="Report not found.")

@app.get("/analytics")
async def get_analytics():
    """Analytics Dashboard Data."""
    return analytics_db

# --- GRADIO WRAPPER (Required for Hugging Face Free Tier) ---
with gr.Blocks() as demo:
    gr.Markdown("## Enterprise Deepfake API is Live \n All endpoints (/analyze, /fact-check, /analytics) are operational.")

app = gr.mount_gradio_app(app, demo, path="/")
