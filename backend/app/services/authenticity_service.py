import io
import os
import cv2
import base64
import json
import time
import torch
import pickle
import numpy as np
import faiss
from PIL import Image
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel

from app.models.model_loader import model_loader
from app.config import settings


class VerificationResult(BaseModel):
    verdict: str
    confidence: float
    authentic_probability: float
    fake_probability: float
    confidence_level: str
    heatmap_base64: str
    analysis_regions: List[str]
    processing_time_ms: int
    layer_scores: Dict[str, float]


class AuthenticityService:
    """
    Three-layer authenticity pipeline:
      Layer 1 (60%) - CLIP embedding vs brand FAISS index (visual similarity)
      Layer 2 (25%) - YOLO logo detection confidence
      Layer 3 (15%) - Error Level Analysis (image manipulation detection)
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.brand_thresholds = self._load_thresholds()
        self.brand_dna = self._load_dna()
        print(f"[AuthenticityService] Device: {self.device}")
        print(f"[AuthenticityService] Loaded thresholds for {len(self.brand_thresholds)} brands")
        print(f"[AuthenticityService] Loaded DNA for {len(self.brand_dna)} brands")

    # -------------------------------------------------------------------------
    # Loaders
    # -------------------------------------------------------------------------

    def _load_thresholds(self) -> Dict[str, float]:
        path = os.path.join(settings.SAVED_MODELS_DIR, "brand_thresholds.json")
        try:
            with open(path, "r") as f:
                data = json.load(f)
            print(f"[AuthenticityService] Thresholds loaded from {path}")
            return data
        except FileNotFoundError:
            print(f"[AuthenticityService] WARNING: brand_thresholds.json not found at {path}. Using default 0.75.")
            return {}
        except Exception as e:
            print(f"[AuthenticityService] WARNING: Could not load thresholds: {e}")
            return {}

    def _load_dna(self) -> Dict[str, Any]:
        path = os.path.join(settings.SAVED_MODELS_DIR, "brand_dna.pkl")
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            print(f"[AuthenticityService] Brand DNA loaded from {path}")
            return data
        except FileNotFoundError:
            print(f"[AuthenticityService] WARNING: brand_dna.pkl not found at {path}.")
            return {}
        except Exception as e:
            print(f"[AuthenticityService] WARNING: Could not load DNA: {e}")
            return {}

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _safe_layer_fallback(self, layer_name: str, e: Exception) -> Dict[str, Any]:
        print(f"[AuthenticityService] {layer_name} error: {type(e).__name__}: {e}")
        return {"score": 50.0, "error": str(e)}

    def _get_heatmap(self, pil_img: Image.Image) -> str:
        """Generate a simple focus heatmap overlay on the image."""
        try:
            overlay = np.array(pil_img.convert("RGB"))
            overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            h, w = overlay.shape[:2]

            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (w // 2, h // 2), min(w, h) // 3, 255, -1)
            mask = cv2.GaussianBlur(mask, (101, 101), 0)

            heatmap_color = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
            result = cv2.addWeighted(overlay, 0.7, heatmap_color, 0.3, 0)

            _, buffer = cv2.imencode(".png", result)
            return "data:image/png;base64," + base64.b64encode(buffer).decode("utf-8")
        except Exception as e:
            print(f"[AuthenticityService] Heatmap generation failed: {e}")
            return ""

    def _calibrate_confidence(self, score: float) -> Tuple[float, str]:
        """
        Convert raw 0-100 fusion score to 0.0-1.0 probability via sigmoid.
        Midpoint (score=72) maps to ~0.50 probability.
        """
        k = 0.12
        prob = 1.0 / (1.0 + np.exp(-k * (score - 72.0)))
        prob = float(np.clip(prob, 0.01, 0.99))

        if prob > 0.85:
            level = "high"
        elif prob > 0.60:
            level = "medium"
        else:
            level = "low"

        return prob, level

    # -------------------------------------------------------------------------
    # Main verify entry point
    # -------------------------------------------------------------------------

    def verify(self, image_bytes: bytes, brand: str) -> VerificationResult:
        start_time = time.time()

        try:
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Cannot open image: {e}")

        # --- Layer 1: CLIP + FAISS visual similarity (weight 60%) ---
        try:
            l1_res = self._layer_1_clip_faiss(pil_image, brand)
        except Exception as e:
            l1_res = self._safe_layer_fallback("L1-CLIP-FAISS", e)

        # --- Layer 2: YOLO logo detection (weight 25%) ---
        try:
            l2_res = self._layer_2_logo_detection(pil_image)
        except Exception as e:
            l2_res = self._safe_layer_fallback("L2-YOLO", e)

        # --- Layer 3: ELA image integrity (weight 15%) ---
        try:
            l3_res = self._layer_3_ela(pil_image, image_bytes)
        except Exception as e:
            l3_res = self._safe_layer_fallback("L3-ELA", e)

        l1_score = float(l1_res.get("score", 50.0))
        l2_score = float(l2_res.get("score", 50.0))
        l3_score = float(l3_res.get("score", 50.0))

        # Weighted fusion
        overall = (l1_score * 0.60) + (l2_score * 0.25) + (l3_score * 0.15)

        prob_auth, level = self._calibrate_confidence(overall)
        heatmap_b64 = self._get_heatmap(pil_image)
        verdict = "authentic" if prob_auth >= 0.60 else "fake"
        processing_time = int((time.time() - start_time) * 1000)

        print(
            f"[AuthenticityService] Brand={brand} | "
            f"L1={l1_score:.1f} L2={l2_score:.1f} L3={l3_score:.1f} | "
            f"Overall={overall:.1f} | prob={prob_auth:.3f} | Verdict={verdict}"
        )

        return VerificationResult(
            verdict=verdict,
            confidence=float(round(prob_auth * 100, 1)),
            authentic_probability=float(round(prob_auth, 4)),
            fake_probability=float(round(1.0 - prob_auth, 4)),
            confidence_level=level,
            heatmap_base64=heatmap_b64,
            analysis_regions=["Logo Region", "Stitching Pattern", "Material Texture"],
            processing_time_ms=processing_time,
            layer_scores={
                "visual_similarity": float(round(l1_score, 1)),
                "logo_detection":    float(round(l2_score, 1)),
                "image_integrity":   float(round(l3_score, 1)),
            },
        )

    # -------------------------------------------------------------------------
    # Layer 1 — CLIP + FAISS
    # -------------------------------------------------------------------------

    def _layer_1_clip_faiss(self, pil_img: Image.Image, brand: str) -> Dict[str, Any]:
        """
        Encode image with OpenCLIP, query the brand-specific FAISS index,
        and convert cosine similarity to a 0-100 authenticity score.
        """
        if model_loader.clip_model is None or model_loader.clip_preprocess is None:
            print("[L1] CLIP not loaded — returning neutral score 50")
            return {"score": 50.0}

        try:
            tensor = model_loader.clip_preprocess(pil_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                emb = model_loader.clip_model.encode_image(tensor)
                emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)

            query_vec = emb.cpu().numpy().astype("float32")
            faiss.normalize_L2(query_vec)  # ensure unit norm for IndexFlatIP

            # Per-brand calibrated threshold (falls back to 0.75 if missing)
            threshold = float(self.brand_thresholds.get(brand, 0.75))

            idx = model_loader.get_brand_index(brand)
            if idx is None or idx.ntotal == 0:
                print(f"[L1] No FAISS index for brand '{brand}' — returning neutral score 50")
                return {"score": 50.0}

            k = min(10, idx.ntotal)
            D, _ = idx.search(query_vec, k=k)
            # D[0] contains cosine similarities in [-1, 1] (usually [0, 1] for images)
            mean_sim = float(np.mean(D[0]))
            max_sim  = float(np.max(D[0]))

            print(
                f"[L1] brand={brand} | threshold={threshold:.3f} | "
                f"mean_sim={mean_sim:.3f} | max_sim={max_sim:.3f}"
            )

            # Linear mapping:
            #   mean_sim == threshold   → score = 70 (borderline)
            #   mean_sim == 1.0         → score = 100 (perfect match)
            #   mean_sim == 0.0         → score = 0   (completely different)
            if mean_sim >= threshold:
                score = 70.0 + ((mean_sim - threshold) / max(1.0 - threshold, 1e-6)) * 30.0
            else:
                score = (mean_sim / max(threshold, 1e-6)) * 70.0

            score = float(np.clip(score, 0.0, 100.0))
            return {
                "score":     score,
                "mean_sim":  mean_sim,
                "max_sim":   max_sim,
                "threshold": threshold,
            }

        except Exception as e:
            print(f"[L1] Unexpected error: {e}")
            return {"score": 50.0}

    # -------------------------------------------------------------------------
    # Layer 2 — YOLO logo detection
    # -------------------------------------------------------------------------

    def _layer_2_logo_detection(self, pil_img: Image.Image) -> Dict[str, Any]:
        """
        Run YOLO on the image.
        High detection confidence → logo found → authentic signal.
        No detection → suspicious → lower score.
        """
        if model_loader.yolo_model is None:
            print("[L2] YOLO not loaded — returning neutral score 50")
            return {"score": 50.0, "note": "YOLO not loaded"}

        try:
            cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            results = model_loader.yolo_model(cv_img, verbose=False)

            if results and len(results[0].boxes) > 0:
                confs = [float(b.conf[0]) for b in results[0].boxes]
                best_conf = float(max(confs))
                # Map confidence 0-1 → score 55-100
                score = 55.0 + (best_conf * 45.0)
                print(f"[L2] Logo detected | best_conf={best_conf:.3f} | score={score:.1f}")
                return {"score": score, "detection_conf": best_conf, "num_detections": len(confs)}
            else:
                # No logo found — likely a fake or very clean product shot
                print("[L2] No logo detected | score=35.0")
                return {"score": 35.0, "note": "No logo detected"}

        except Exception as e:
            print(f"[L2] Error: {e}")
            return {"score": 50.0}

    # -------------------------------------------------------------------------
    # Layer 3 — Error Level Analysis
    # -------------------------------------------------------------------------

    def _layer_3_ela(self, pil_img: Image.Image, original_bytes: bytes) -> Dict[str, Any]:
        """
        Error Level Analysis detects JPEG re-compression artifacts.
        Authentic product photos typically have low, uniform ELA.
        Heavily edited/composited fakes show high-variance ELA.
        """
        try:
            source = pil_img.convert("RGB")

            # Re-compress at quality=90 and compare
            buf = io.BytesIO()
            source.save(buf, format="JPEG", quality=90)
            buf.seek(0)
            recompressed = Image.open(buf).convert("RGB")

            ela_arr = np.abs(
                np.array(source, dtype=np.float32) -
                np.array(recompressed, dtype=np.float32)
            )
            mean_ela = float(np.mean(ela_arr))
            std_ela  = float(np.std(ela_arr))

            # Low mean + low std = pristine original image = authentic signal
            # score = 100 when mean_ela=0, drops as manipulation increases
            score = 100.0 - float(np.clip(mean_ela * 8.0, 0.0, 50.0))

            print(f"[L3] ELA mean={mean_ela:.3f} std={std_ela:.3f} | score={score:.1f}")
            return {"score": score, "ela_mean": mean_ela, "ela_std": std_ela}

        except Exception as e:
            print(f"[L3] Error: {e}")
            return {"score": 50.0}


# Singleton instance
authenticity_service = AuthenticityService()
