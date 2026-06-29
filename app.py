from __future__ import annotations
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd, streamlit as st, matplotlib.pyplot as plt
from src.data import make_synthetic, TARGET_NAME
from src.model import train_all_models, cross_validate
from src.visualizations import *
st.set_page_config(page_title="IDVerify | Jumio Identity Verification", layout="wide", page_icon="\U0001f9ec")
with st.sidebar:
    st.header("\u2699 Config"); n=st.slider("Samples",2000,20000,10000,1000); tau=st.slider("Threshold",0.05,0.95,0.50,0.05)
    st.caption("Jumio | Identity Verification & AML/KYC")
data=make_synthetic(n=n); b=train_all_models(data)
y_test=b["y_test"]; y_probas={n:b["results"][n]["y_proba"] for n in b["results"]}
best=max(b["results"],key=lambda n: b["results"][n]["metrics"].get("roc_auc",0))
c1,c2,c3,c4=st.columns(4)
c1.metric("Samples",f"{n:,}"); c2.metric("Verification Rate",f"{data['positive_rate']:.1%}")
c3.metric("Best AUC",f"{b['results'][best]['metrics']['roc_auc']:.4f}"); c4.metric("Best",best)
t1,t2,t3,t4=st.tabs(["\U0001f4ca Explorer","\U0001f52c Model Lab","\U0001f50d Fraud Analysis","\U0001f3af Liveness"])
with t1:
    st.dataframe(data["df"].head(50),use_container_width=True,height=200)
    fig,ax=plt.subplots(figsize=(5,3)); _style()
    ax.bar(["Rejected","Verified"],[1-data["positive_rate"],data["positive_rate"]],color=["#f43f5e","#22c55e"])
    for i,v in enumerate([1-data["positive_rate"],data["positive_rate"]]): ax.text(i,v+.01,f"{v:.1%}",ha="center",color="white")
    ax.set_title("Verification Outcome Distribution",color="white"); ax.grid(True,alpha=.2)
    st.pyplot(fig)
with t2:
    rows=[{**{"Model":n},**{k:f"{v:.4f}" for k,v in r["metrics"].items() if k!="confusion_matrix"}} for n,r in b["results"].items()]
    st.dataframe(pd.DataFrame(rows).set_index("Model"),use_container_width=True)
    col_a,col_b=st.columns(2)
    with col_a: st.pyplot(plot_roc_curve(y_test,y_probas))
    with col_b: st.pyplot(plot_calibration_curve(y_test,y_probas))
    st.pyplot(plot_confusion_matrix(y_test,b["results"]["XGBoost"]["y_pred"],"XGBoost"))
    cv=cross_validate(data); cvr=[{"Model":n,"AUC":f"{s['roc_auc']['mean']:.4f}","\u00b1":f"\u00b1{s['roc_auc']['std']:.4f}"} for n,s in cv.items()]
    st.dataframe(pd.DataFrame(cvr).set_index("Model"),use_container_width=True)
with t3:
    st.subheader("Document Authenticity Analysis")
    auth_map={"genuine":"Genuine","forged":"Forged","altered":"Altered","suspicious":"Suspicious"}
    data["df"]["auth_label"]=data["df"]["document_authenticity"].map(auth_map)
    auth_dist=data["df"]["auth_label"].value_counts()
    fig,ax=plt.subplots(figsize=(6,4)); _style()
    colors=["#22c55e","#f43f5e","#f97316","#fbbf24"]
    ax.pie(auth_dist.values,labels=auth_dist.index,autopct="%1.1f%%",colors=colors,textprops={"color":"white"})
    ax.set_title("Document Authenticity Distribution",color="white")
    st.pyplot(fig)
    col_a,col_b=st.columns(2)
    with col_a:
        fig,ax=plt.subplots(figsize=(5,4)); _style()
        verified=data["df"][data["df"]["verified"]==1]["face_match_score"]
        rejected=data["df"][data["df"]["verified"]==0]["face_match_score"]
        ax.hist(verified,bins=30,alpha=0.5,color="#22c55e",label="Verified",density=True)
        ax.hist(rejected,bins=30,alpha=0.5,color="#f43f5e",label="Rejected",density=True)
        ax.set_title("Face Match Score",color="white"); ax.legend(fontsize=8); ax.grid(True,alpha=.2)
        st.pyplot(fig)
    with col_b:
        fig,ax=plt.subplots(figsize=(5,4)); _style()
        verified=data["df"][data["df"]["verified"]==1]["liveness_score"]
        rejected=data["df"][data["df"]["verified"]==0]["liveness_score"]
        ax.hist(verified,bins=30,alpha=0.5,color="#22c55e",label="Verified",density=True)
        ax.hist(rejected,bins=30,alpha=0.5,color="#f43f5e",label="Rejected",density=True)
        ax.set_title("Liveness Score",color="white"); ax.legend(fontsize=8); ax.grid(True,alpha=.2)
        st.pyplot(fig)
with t4:
    st.subheader("Liveness Detection Analysis")
    st.latex(r"\text{Liveness} = \sigma(w_1 \cdot s_{\text{face}} + w_2 \cdot s_{\text{microprint}} + w_3 \cdot s_{\text{UV}})")
    live_good=data["df"][data["df"]["verified"]==1]["liveness_score"]
    live_bad=data["df"][data["df"]["verified"]==0]["liveness_score"]
    col_a,col_b=st.columns(2)
    with col_a:
        fig,ax=plt.subplots(figsize=(5,4)); _style()
        ax.hist(live_good,bins=30,alpha=0.5,color="#22c55e",label="Verified",density=True)
        ax.hist(live_bad,bins=30,alpha=0.5,color="#f43f5e",label="Rejected",density=True)
        ax.set_title("Liveness Score Distribution",color="white"); ax.legend(fontsize=8); ax.grid(True,alpha=.2)
        st.pyplot(fig)
    with col_b:
        fig,ax=plt.subplots(figsize=(5,4)); _style()
        ax.hist(data["df"]["microprint_score"],bins=30,color="#22d3ee",alpha=0.6,edgecolor="#1a1f2e")
        ax.axvline(data["df"]["microprint_score"].mean(),color="#f97316",ls="--",lw=2,label=f"Mean={data['df']['microprint_score'].mean():.3f}")
        ax.set_title("Microprint Score Distribution",color="white"); ax.legend(fontsize=8); ax.grid(True,alpha=.2)
        st.pyplot(fig)
    st.subheader("MRZ Confidence by Document Type")
    mrz_auth=data["df"].groupby("document_authenticity")["mrz_confidence"].mean()
    fig,ax=plt.subplots(figsize=(5,3)); _style()
    ax.bar(range(4),mrz_auth.values,color=["#22c55e","#f43f5e","#f97316","#fbbf24"])
    ax.set_xticks(range(4)); ax.set_xticklabels([auth_map[k] for k in mrz_auth.index]); ax.set_title("Avg MRZ Confidence",color="white"); ax.grid(True,alpha=.2)
    st.pyplot(fig)
