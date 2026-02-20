import torch
import numpy as np
import librosa
import io
import requests
from transformers import Wav2Vec2FeatureExtractor, AutoModelForAudioClassification
import torch.nn.functional as F
from fastapi import HTTPException


class VoiceDetector:
    _instance = None

    def __init__(self):
        from torch.nn import CosineSimilarity
        self.cos_sim = CosineSimilarity(dim=1, eps=1e-6)

        self.detector_model_name = "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.detector_model_name
        )
        self.model = AutoModelForAudioClassification.from_pretrained(
            self.detector_model_name
        )
        self.model.eval()

        self.whisper_model_name = None
        self.transcriber = None
        self.fraud_keywords = []

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_audio(self, input_audio):
        """
        Download or decode the audio from URL or Base64/Bytes.
        Returns floating point audio array and sample rate.
        """
        if isinstance(input_audio, str) and input_audio.startswith("http"):
            response = requests.get(input_audio)
            response.raise_for_status()
            audio_bytes = io.BytesIO(response.content)
        elif isinstance(input_audio, (bytes, bytearray, io.BytesIO)):
            if isinstance(input_audio, (bytes, bytearray)):
                audio_bytes = io.BytesIO(input_audio)
            else:
                audio_bytes = input_audio
        elif isinstance(input_audio, np.ndarray):
            return input_audio, 16000
        else:
            audio_bytes = input_audio

        try:
            y, sr = librosa.load(audio_bytes, sr=None)
            return y, sr
        except Exception as e:
            raise ValueError(f"Failed to load audio: {e}")

    def _preprocess_audio(self, y, sr):
        """
        Convert to mono, 16 kHz.
        Apply silence trimming and normalization to -1..1.
        """
        target_sr = 16000

        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

        if y.ndim > 1:
            y = librosa.to_mono(y)

        y, _ = librosa.effects.trim(y)

        max_val = np.abs(y).max()
        if max_val > 0:
            y = y / max_val

        return y, target_sr

    def _chunk_audio(self, y, sr, chunk_duration=30):
        """Split audio into chunks if longer than chunk_duration seconds."""
        duration = len(y) / sr
        chunks = []
        if duration > chunk_duration:
            samples_per_chunk = int(chunk_duration * sr)
            total_samples = len(y)
            for start in range(0, total_samples, samples_per_chunk):
                end = min(start + samples_per_chunk, total_samples)
                chunks.append(y[start:end])
        else:
            chunks.append(y)
        return chunks

    def _calculate_smoothness(self, embeddings: torch.Tensor) -> float:
        """
        Calculates temporal smoothness.
        AI voices tend to have higher frame-to-frame cosine similarity.
        """
        if embeddings.shape[1] < 2:
            return 0.0
        similarity = self.cos_sim(embeddings[0, :-1, :], embeddings[0, 1:, :])
        return float(similarity.mean().item())

    def _calculate_snr(self, y: np.ndarray) -> float:
        """
        Calculates Signal-to-Noise Ratio (SNR) of the audio.
        High SNR (> 60dB) -> likely AI or studio recording.
        Lower SNR (< 30dB) -> natural background noise (likely Human).
        """
        if len(y) < 100:
            return 0.0

        frame_length = 2048
        hop_length = 512
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)

        sorted_db = np.sort(rms_db[0])

        noise_idx = max(1, int(len(sorted_db) * 0.1))
        noise_floor_db = np.mean(sorted_db[:noise_idx])

        signal_idx = int(len(sorted_db) * 0.8)
        signal_power_db = np.mean(sorted_db[signal_idx:])

        snr_value = signal_power_db - noise_floor_db
        return float(snr_value)

    def _calculate_pitch_score(self, y, sr):
        """
        Estimates 'Human-ness' based on Pitch (F0) variance and jitter.
        Real voices have higher pitch standard deviation and frame-to-frame jitter.
        Returns (score, pitch_std, jitter) where score is 0.0 (Robotic) to 1.0 (Very Human).
        """
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=1000, sr=sr)
            f0 = f0[~np.isnan(f0)]

            if len(f0) < 10:
                return 0.0, 0.0, 0.0

            pitch_std = np.std(f0)
            jitter = np.mean(np.abs(np.diff(f0)))

            # Jitter scoring: Human jitter sweet spot is 2-8 Hz
            if jitter < 1.0:
                score_jitter = 0.0
            elif 1.0 <= jitter < 2.0:
                score_jitter = (jitter - 1.0)
            elif 2.0 <= jitter <= 8.0:
                score_jitter = 1.0
            elif 8.0 < jitter < 12.0:
                score_jitter = 1.0 - ((jitter - 8.0) / 4.0)
            else:
                score_jitter = 0.0

            # Pitch std scoring
            if pitch_std < 5.0:
                score_std = 0.0
            else:
                score_std = min(1.0, pitch_std / 20.0)

            # Weight jitter higher (80%) because intonation is easier to fake
            final_score = (score_std * 0.2) + (score_jitter * 0.8)

            return final_score, pitch_std, jitter
        except Exception:
            return 0.0, 0.0, 0.0

    def detect_fraud(self, input_audio, metadata=None):
        """
        Main detection pipeline. Combines deep learning model with
        acoustic heuristics and metadata forensics.
        """
        smoothness = 0.0
        time_variance = 0.0
        heuristic_score = 0.0
        probs = None
        pitch_score = 0.0
        snr_score = 0.0
        metadata_hit = False
        metadata_explanation = ""
        metadata_note = None

        # --- Metadata Forensics ---
        if metadata:
            encoder = metadata.get("encoder", "").lower()
            # Programmatic encoders (lavf/lavc = FFmpeg, google = TTS API)
            # Real recordings typically use iTunes, Android, or no encoder tag
            if "lavf" in encoder or "lavc" in encoder or "google" in encoder:
                metadata_hit = True
                metadata_explanation = f"Metadata analysis detected programmatic encoder: {metadata.get('encoder')}"

        # --- Audio Loading & Preprocessing ---
        raw_y, raw_sr = self._load_audio(input_audio)
        y, sr = self._preprocess_audio(raw_y, raw_sr)
        if y is None or y.size == 0:
            raise HTTPException(status_code=400, detail="Decoded audio contained no samples after preprocessing")

        # Limit to 4 seconds for optimal accuracy/speed tradeoff
        max_samples = 16000 * 4
        if len(y) > max_samples:
            y = y[:max_samples]

        chunks = [c for c in self._chunk_audio(y, sr) if len(c) > 0]
        if not chunks:
            raise HTTPException(status_code=400, detail="Audio contained no decodable frames")

        # --- Primary AI vs Human Detection (Wav2Vec2) ---
        ai_probs = []
        all_embeddings = []

        for chunk in chunks:
            inputs = self.feature_extractor(
                chunk,
                sampling_rate=sr,
                return_tensors="pt",
                padding=True
            )

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                logits = outputs.logits

                if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                    embeddings = outputs.hidden_states[-1]
                    all_embeddings.append(embeddings)

            probs = F.softmax(logits, dim=-1)
            p_ai_chunk = probs[0][1].item()
            ai_probs.append(p_ai_chunk)

        p_ai_model = sum(ai_probs) / len(ai_probs) if ai_probs else 0.0

        # --- Acoustic Heuristics ---

        # 1. Pitch Jitter (Human voices have natural instability)
        pitch_score, p_std, p_jitter = self._calculate_pitch_score(y, sr)

        # 2. SNR (AI voices often have unnaturally clean audio)
        snr_val = self._calculate_snr(y)
        if snr_val > 40:
            snr_score = min(1.0, (snr_val - 40) / 20.0)
        elif snr_val < 30:
            snr_score = -min(1.0, (30 - snr_val) / 20.0)
        else:
            snr_score = 0.0

        # 3. Temporal Smoothness (AI voices are too consistent)
        if all_embeddings:
            embeddings_tensor = all_embeddings[0]
            smoothness = self._calculate_smoothness(embeddings_tensor)

        # 4. Combined heuristic score
        heuristic_score = (
            (1.0 - pitch_score) * 0.5 +
            smoothness * 0.3 +
            max(0, snr_score) * 0.2
        )

        time_variance = 1.0 - smoothness

        # --- Heuristic Adjustments ---
        final_p_ai = p_ai_model

        # Boost AI probability if heuristics strongly suggest AI
        if heuristic_score > 0.6:
            ai_boost = heuristic_score * 0.15
            final_p_ai = min(0.98, final_p_ai + ai_boost)

        # Boost HUMAN probability if strong human pitch characteristics
        if pitch_score > 0.7 and p_ai_model < 0.5:
            human_boost = pitch_score * 0.15
            final_p_ai = max(0.02, final_p_ai - human_boost)

        # SNR adjustment
        if snr_score < -0.5:
            noise_boost = abs(snr_score) * 0.08
            final_p_ai = max(0.02, final_p_ai - noise_boost)
        elif snr_score > 0.5:
            clean_boost = snr_score * 0.08
            final_p_ai = min(0.98, final_p_ai + clean_boost)

        # --- Metadata as Strong Signal ---
        if metadata_hit:
            metadata_note = metadata_explanation or "Suspicious encoder metadata detected"
            if 0.35 <= final_p_ai <= 0.65:
                # Borderline case: metadata is strong evidence of AI
                final_p_ai = 0.75
            elif final_p_ai > 0.65:
                final_p_ai = min(0.98, final_p_ai + 0.1)
            elif final_p_ai < 0.35:
                # Model confident it's human but metadata suggests programmatic encoding
                # Use moderate boost - real humans can be re-encoded with FFmpeg
                final_p_ai = min(0.50, final_p_ai + 0.15)

        # --- Confidence Amplification ---
        AI_THRESHOLD = 0.48

        if final_p_ai > AI_THRESHOLD:
            # AI prediction: map [0.48, 1.0] to [0.85, 0.98]
            distance_from_threshold = final_p_ai - AI_THRESHOLD
            normalized = distance_from_threshold / (1.0 - AI_THRESHOLD)
            amplified = 0.85 + (normalized ** 0.7) * 0.13
            final_p_ai = min(0.98, amplified)
        else:
            # Human prediction: map [0.0, 0.48] to [0.02, 0.15]
            if final_p_ai < 0.25:
                amplified = final_p_ai * 0.3
                final_p_ai = max(0.02, amplified)
            else:
                normalized = (AI_THRESHOLD - final_p_ai) / AI_THRESHOLD
                final_p_ai = 0.15 - (normalized ** 0.7) * 0.13

        # Final classification
        classification = "AI" if final_p_ai > 0.5 else "Human"
        confidence = max(final_p_ai, 1 - final_p_ai)
        p_ai = final_p_ai

        # --- Build Explanation ---
        parts = [
            f"AI probability {round(p_ai, 2)}",
            f"Deepfake detector classified as {classification}"
        ]
        if metadata_note:
            parts.append(metadata_note)
        if heuristic_score > 0.5:
            parts.append("Robotic voice patterns detected")
        elif pitch_score > 0.75:
            parts.append("Natural human pitch variations detected")
        if snr_score < 0:
            parts.append("Natural background noise detected")
        elif snr_score > 0:
            parts.append("Studio-quality silence detected")

        explanation = ", ".join(parts)
        audio_duration_seconds = round(len(y) / sr, 2)

        return {
            "classification": classification,
            "confidence_score": round(confidence, 2),
            "ai_probability": round(p_ai, 2),
            "detected_language": "N/A",
            "transcription": "Disabled for performance",
            "english_translation": "Disabled for performance",
            "fraud_keywords": [],
            "overall_risk": "LOW",
            "explanation": explanation,
            "audio_duration_seconds": audio_duration_seconds,
            "num_chunks_processed": len(chunks),
            "chunk_ai_probabilities": [round(p, 3) for p in ai_probs],
            "heuristic_score": round(heuristic_score, 3),
            "pitch_human_score": round(pitch_score, 3),
            "pitch_std": round(p_std, 2),
            "pitch_jitter": round(p_jitter, 2),
            "smoothness_score": round(smoothness, 4),
            "variance_score": round(time_variance, 5),
            "snr_score": round(snr_val, 2) if 'snr_val' in locals() else 0.0,
            "debug_probs": [round(p, 4) for p in probs[0].tolist()] if probs is not None else [],
            "debug_labels": self.model.config.id2label if self.model.config.id2label else "None"
        }


# Global instance
detector = None

def get_detector():
    global detector
    if detector is None:
        detector = VoiceDetector.get_instance()
    return detector
