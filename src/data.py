from __future__ import annotations
import numpy as np; import pandas as pd
FEATURE_NAMES = ["face_match_score","liveness_score","document_authenticity","text_field_consistency","photo_quality","edge_detection_score","qr_barcode_valid","mrz_confidence","template_match","font_analysis","microprint_score","uv_feature_score"]
CATEGORICAL_FEATURES = ["document_authenticity"]
NUMERICAL_FEATURES = ["face_match_score","liveness_score","text_field_consistency","photo_quality","edge_detection_score","qr_barcode_valid","mrz_confidence","template_match","font_analysis","microprint_score","uv_feature_score"]
TARGET_NAME = "verified"
def make_synthetic(n=10000,seed=42):
    rng=np.random.default_rng(seed)
    df=pd.DataFrame({
        "face_match_score": rng.beta(7,2,size=n).round(3),
        "liveness_score": rng.beta(6,2,size=n).round(3),
        "document_authenticity": rng.choice(["genuine","forged","altered","suspicious"],size=n,p=[0.75,0.08,0.10,0.07]),
        "text_field_consistency": rng.beta(8,3,size=n).round(3),
        "photo_quality": rng.uniform(0,100,size=n).round(1),
        "edge_detection_score": rng.beta(5,3,size=n).round(3),
        "qr_barcode_valid": rng.choice([0,1],size=n,p=[0.15,0.85]),
        "mrz_confidence": rng.beta(9,2,size=n).round(3),
        "template_match": rng.uniform(0,100,size=n).round(1),
        "font_analysis": rng.beta(7,3,size=n).round(3),
        "microprint_score": rng.beta(5,2,size=n).round(3),
        "uv_feature_score": rng.beta(6,3,size=n).round(3),
    })
    face=df["face_match_score"]; live=df["liveness_score"]; auth=df["document_authenticity"].map({"genuine":0,"altered":0.5,"suspicious":0.7,"forged":1}).values
    text=df["text_field_consistency"]; photo=df["photo_quality"]/100; edge=df["edge_detection_score"]
    qr=df["qr_barcode_valid"]; mrz=df["mrz_confidence"]; tmpl=df["template_match"]/100
    font=df["font_analysis"]; micro=df["microprint_score"]; uv=df["uv_feature_score"]
    log_odds = 4.0 + 1.0*face + 0.8*live - 1.2*auth + 0.5*text + 0.3*photo + 0.2*edge + 0.4*qr + 0.5*mrz + 0.3*tmpl + 0.2*font + 0.3*micro + 0.2*uv + rng.normal(0,0.4,size=n)
    prob=1/(1+np.exp(-log_odds)); y=(prob>np.percentile(prob,70)).astype(np.float64)
    return {"X":df,"y":y,"features":FEATURE_NAMES,"df":df.assign(verified=y),"categorical_features":CATEGORICAL_FEATURES,"numerical_features":NUMERICAL_FEATURES,"n_samples":n,"n_features":len(FEATURE_NAMES),"positive_rate":y.mean()}
