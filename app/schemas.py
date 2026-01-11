from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class MetadataLayer(BaseModel):
    status: str  # "verified_ai", "verified_human", "not_found"
    provider: Optional[str] = None
    description: Optional[str] = "Validation performed via file metadata."
    # Optional debug/telemetry fields
    human_score: Optional[float] = None
    ai_score: Optional[float] = None
    signals: Optional[List[str]] = None

class ForensicsLayer(BaseModel):
    status: str  # "detected", "not_detected", "in_progress", "skipped"
    probability: Optional[float] = 0.0
    signals: Optional[List[str]] = []

class DetectionLayers(BaseModel):
    layer1_metadata: MetadataLayer
    layer2_forensics: Optional[ForensicsLayer] = None

class MetadataSummary(BaseModel):
    human_score: float
    ai_score: float
    signals: List[str]
    extracted: Optional[Dict[str, Any]] = None
    bypass_reason: Optional[str] = None

class DetectionResponse(BaseModel):
    summary: str
    confidence_score: float
    suspicious: bool = False
    layers: DetectionLayers
    gpu_time_ms: Optional[float] = 0.0
    gpu_bypassed: Optional[bool] = False
    metadata: Optional[MetadataSummary] = None