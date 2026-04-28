"""Dry-run completo do clustering. Reproduz o estado pós-EDA do notebook e roda os algoritmos."""
import warnings, time
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, pair_confusion_matrix,
)

DATA_DIR = Path("data/diabetes+130-us+hospitals+for+years+1999-2008")
RNG = 42

# =============================================================================
# REPRODUZ O PIPELINE DE EDA/PREPROC (compactado, sem prints)
# =============================================================================
df = pd.read_csv(DATA_DIR / "diabetic_data.csv", low_memory=False)
df.replace("?", np.nan, inplace=True)

EXPIRED_HOSPICE = {11, 13, 14, 19, 20, 21}
df = df[~df["discharge_disposition_id"].isin(EXPIRED_HOSPICE)].copy()
df.drop(columns=["encounter_id"], inplace=True)
df.sort_values("patient_nbr", inplace=True)
df = df.drop_duplicates("patient_nbr", keep="first")
df.drop(columns=["patient_nbr", "weight", "payer_code"], inplace=True, errors="ignore")

ADM_TYPE = {1:"Emergency",2:"Urgent",3:"Elective",4:"Newborn",5:"Unknown",6:"Unknown",7:"Trauma",8:"Unknown"}
DISCH = {1:"Home",6:"HomeHealth",8:"HomeHealth",3:"SNF",4:"ICF",5:"OtherInpatient",2:"TransferHospital",
         22:"Rehab",23:"LongTerm",24:"SNF",27:"OtherInpatient",28:"Psych",29:"OtherInpatient",7:"AMA",
         9:"Inpatient",10:"Inpatient",12:"Inpatient",15:"OtherInpatient",16:"Outpatient",17:"Outpatient",
         18:"Unknown",25:"Unknown",26:"Unknown",30:"Unknown"}
ADM_SRC = {1:"Referral",2:"Referral",3:"Referral",4:"Transfer",5:"Transfer",6:"Transfer",10:"Transfer",
           18:"Transfer",19:"Transfer",22:"Transfer",25:"Transfer",26:"Transfer",7:"Emergency",8:"Law",
           9:"Unknown",15:"Unknown",17:"Unknown",20:"Unknown",21:"Unknown",
           11:"Birth",12:"Birth",13:"Birth",14:"Birth",23:"Birth",24:"Birth"}
df["admission_type"]   = df["admission_type_id"].map(ADM_TYPE).fillna("Unknown")
df["discharge_to"]     = df["discharge_disposition_id"].map(DISCH).fillna("Unknown")
df["admission_source"] = df["admission_source_id"].map(ADM_SRC).fillna("Unknown")
df.drop(columns=["admission_type_id","discharge_disposition_id","admission_source_id"], inplace=True)

def group_icd9(c):
    if pd.isnull(c): return "Other"
    s = str(c)
    if s.startswith("V") or s.startswith("E"): return "Other"
    try: v = float(c)
    except: return "Other"
    if 390<=v<=459 or v==785: return "Circulatory"
    if 460<=v<=519 or v==786: return "Respiratory"
    if 520<=v<=579 or v==787: return "Digestive"
    if 250<=v<251: return "Diabetes"
    if 800<=v<=999: return "Injury"
    if 710<=v<=739: return "Musculoskeletal"
    if 580<=v<=629 or v==788: return "Genitourinary"
    if 140<=v<=239: return "Neoplasms"
    return "Other"
for c in ["diag_1","diag_2","diag_3"]:
    df[c] = df[c].apply(group_icd9)

df["A1Cresult"] = df["A1Cresult"].fillna("NotMeasured")
df["max_glu_serum"] = df["max_glu_serum"].fillna("NotMeasured")
df["medical_specialty"] = df["medical_specialty"].fillna("Unknown")
df["race"] = df["race"].fillna("Unknown")
df = df[df["gender"] != "Unknown/Invalid"].copy()

def near_constant(s, t=0.99):
    vc = s.value_counts(normalize=True, dropna=False)
    return len(vc)==0 or vc.iloc[0]>=t
near = [c for c in df.columns if c!="readmitted" and near_constant(df[c])]
df.drop(columns=near, inplace=True)

for c in ["number_outpatient","number_emergency","number_inpatient"]:
    df[c] = np.log1p(df[c])
age_order = sorted(df["age"].dropna().unique())
df["age"] = df["age"].map({v:i for i,v in enumerate(age_order)}).astype(float)

df = df.reset_index(drop=True)
y_hidden = df["readmitted"].copy()
X = df.drop(columns=["readmitted"])
cont_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols  = X.select_dtypes(include="object").columns.tolist()

preproc = ColumnTransformer([
    ("num", StandardScaler(), cont_cols),
    ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cat_cols),
])
X_enc = preproc.fit_transform(X)
print(f"[preproc] X_enc={X_enc.shape}, y={y_hidden.shape}")

# FAMD
import prince
X_for_famd = X.copy()
X_for_famd[cont_cols] = StandardScaler().fit_transform(X_for_famd[cont_cols])
famd = prince.FAMD(n_components=20, random_state=RNG, engine="sklearn")
famd.fit(X_for_famd)
X_famd = famd.transform(X_for_famd).values
print(f"[famd] {X_famd.shape}")

# Outlier scenario
iso = IsolationForest(contamination=0.05, random_state=RNG, n_jobs=-1)
mask_keep = iso.fit_predict(X_enc) == 1
X_no_iso  = X_enc[mask_keep]
X_famd_no = X_famd[mask_keep]
y_no_iso  = y_hidden.loc[mask_keep].reset_index(drop=True)
print(f"[no-iso] X={X_no_iso.shape}")

# =============================================================================
# HELPERS DE MÉTRICAS
# =============================================================================
def jaccard_pairwise(y_true, y_pred):
    """Jaccard pareado: a / (a+b+c). pair_confusion_matrix devolve [[TN,FP],[FN,TP]]."""
    cm = pair_confusion_matrix(y_true, y_pred)
    tp, fp, fn = cm[1,1], cm[0,1], cm[1,0]
    return tp / (tp + fp + fn) if (tp+fp+fn) > 0 else np.nan

def connectivity(X, labels, n_neighbors=10, sample=10000, seed=RNG):
    """Conectividade de Handl & Knowles. Quanto MENOR melhor."""
    rng = np.random.default_rng(seed)
    n = len(X)
    if n > sample:
        idx = rng.choice(n, sample, replace=False)
        X_s, lab_s = X[idx], labels[idx]
    else:
        X_s, lab_s = X, labels
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(X_s)
    _, knn = nbrs.kneighbors(X_s)
    weights = 1.0 / np.arange(1, n_neighbors+1)
    score = 0.0
    for i in range(len(X_s)):
        neigh = knn[i, 1:]
        score += weights[lab_s[neigh] != lab_s[i]].sum()
    return score

def intra_var(X, labels):
    """Variância intracluster (SSE) — Σ ‖x − centroide_c‖² sobre todos os pontos."""
    total = 0.0
    for c in np.unique(labels):
        if c == -1:  # ruído (DBSCAN)
            continue
        Xc = X[labels == c]
        center = Xc.mean(axis=0)
        total += ((Xc - center) ** 2).sum()
    return total

def evaluate(X, labels, y_true=None, name="", sample_sil=10000):
    """Devolve dict de métricas. NaN se métrica não aplicável (ex: 1 cluster)."""
    out = {"algo": name, "n_clusters": int(len(set(labels)) - (1 if -1 in labels else 0)),
           "n_noise": int((labels == -1).sum())}
    valid_mask = labels != -1
    Xv, lv = X[valid_mask], labels[valid_mask]
    k = len(set(lv))
    if k < 2 or k > len(Xv) - 1:
        out.update({"silhouette": np.nan, "davies_bouldin": np.nan,
                    "calinski_harabasz": np.nan, "intra_var": np.nan,
                    "connectivity": np.nan})
    else:
        rng = np.random.default_rng(RNG)
        if len(Xv) > sample_sil:
            sub = rng.choice(len(Xv), sample_sil, replace=False)
            out["silhouette"] = silhouette_score(Xv[sub], lv[sub])
        else:
            out["silhouette"] = silhouette_score(Xv, lv)
        out["davies_bouldin"]    = davies_bouldin_score(Xv, lv)
        out["calinski_harabasz"] = calinski_harabasz_score(Xv, lv)
        out["intra_var"]         = intra_var(Xv, lv)
        out["connectivity"]      = connectivity(Xv, lv)
    if y_true is not None and k >= 2:
        yv = pd.Categorical(y_true[valid_mask]).codes
        out["ARI"]     = adjusted_rand_score(yv, lv)
        out["jaccard"] = jaccard_pairwise(yv, lv)
    else:
        out["ARI"] = np.nan
        out["jaccard"] = np.nan
    return out

# =============================================================================
# ALGORITMOS
# =============================================================================
results = []

# K-Means++ sweep no espaço FAMD (mais informativo que X_enc para Euclidiana)
print("\n[KMeans++] sweep K=2..10 sobre X_famd_no")
sweep = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=RNG)
    lab = km.fit_predict(X_famd_no)
    sizes = sorted(pd.Series(lab).value_counts().tolist(), reverse=True)
    m = evaluate(X_famd_no, lab, y_no_iso.values, name=f"KMeans++ k={k}")
    m["sizes"] = sizes
    sweep.append(m)
    print(f"  k={k:2d}  sil={m['silhouette']:.4f}  DB={m['davies_bouldin']:.3f}  "
          f"CH={m['calinski_harabasz']:.0f}  ARI={m['ARI']:.3f}  sizes={sizes}")

# escolher melhor K por silhueta
best = max(sweep, key=lambda r: r["silhouette"] if not np.isnan(r["silhouette"]) else -1)
best_k = int(best["algo"].split("k=")[1])
print(f"  -> best k (silhueta) = {best_k}")
results.append(best)

# K-Means random init para o mesmo k
km_rand = KMeans(n_clusters=best_k, init="random", n_init=10, random_state=RNG)
lab_rand = km_rand.fit_predict(X_famd_no)
results.append(evaluate(X_famd_no, lab_rand, y_no_iso.values, name=f"KMeans random k={best_k}"))

# DBSCAN sobre X_famd_no com eps via k-distance plot
print("\n[DBSCAN] k-distance no X_famd_no, k=10")
nbrs = NearestNeighbors(n_neighbors=10).fit(X_famd_no)
kdist = np.sort(nbrs.kneighbors(X_famd_no)[0][:, -1])
# heurística: usar percentil 95 da curva — joelho aproximado
eps_auto = float(np.percentile(kdist, 95))
print(f"  eps (p95 da k-dist) = {eps_auto:.3f}")
# varrer eps em diferentes percentis da k-distance
for pct in [80, 85, 90, 95]:
    eps_v = float(np.percentile(kdist, pct))
    db = DBSCAN(eps=eps_v, min_samples=10, n_jobs=-1).fit(X_famd_no)
    nc = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    nn = (db.labels_ == -1).sum()
    sizes_top = pd.Series(db.labels_[db.labels_ != -1]).value_counts().head(3).tolist()
    print(f"  eps=p{pct} ({eps_v:.2f})  clusters={nc:3d}  noise={nn:5d}  top3={sizes_top}")
    if 2 <= nc <= 30:
        m = evaluate(X_famd_no, db.labels_, y_no_iso.values, name=f"DBSCAN eps=p{pct}")
        m["sizes_top3"] = sizes_top
        results.append(m)

# Hierárquico — em X_enc (o-h padronizado, mais homogêneo) e na amostra estratificada
# Razão: FAMD concentra outliers que dominam single/complete/average e degeneram clusters.
print("\n[Hierárquico] amostra estratificada n=5000 sobre X_enc_no (pós-IsoForest)")
X_enc_no = X_enc[mask_keep]
df_s = pd.DataFrame(X_enc_no).assign(_y=y_no_iso.values)
sample = df_s.groupby("_y", group_keys=False).apply(
    lambda g: g.sample(min(len(g), 5000//y_no_iso.nunique()), random_state=RNG)
)
X_hier = sample.drop(columns="_y").values
y_hier = sample["_y"].values
print(f"  amostra shape: {X_hier.shape}")

for linkage in ["single", "complete", "average"]:
    t0 = time.time()
    ag = AgglomerativeClustering(n_clusters=best_k, linkage=linkage)
    lab = ag.fit_predict(X_hier)
    dt = time.time() - t0
    sizes = pd.Series(lab).value_counts().sort_index().to_dict()
    m = evaluate(X_hier, lab, pd.Series(y_hier), name=f"Agglo {linkage} k={best_k}")
    m["t_sec"] = round(dt, 1)
    m["sizes"] = sizes
    results.append(m)
    print(f"  {linkage:9s}  sil={m['silhouette']:.4f}  DB={m['davies_bouldin']:.3f}  "
          f"ARI={m['ARI']:.3f}  sizes={sizes}  ({dt:.1f}s)")

# Diagnóstico: hierárquico está degenerado? testar k variável e ver se quebra
print("\n[diag hierárquico] k=2..6 com average linkage para ver progressão")
for k in [2,3,4,5,6,8,10]:
    ag = AgglomerativeClustering(n_clusters=k, linkage="average")
    lab = ag.fit_predict(X_hier)
    sizes = pd.Series(lab).value_counts().sort_values(ascending=False).head(5).tolist()
    sil = silhouette_score(X_hier[:3000], lab[:3000]) if k>=2 else np.nan
    print(f"  k={k:2d}  top5 sizes={sizes}  sil={sil:.3f}")

# =============================================================================
# RESUMO
# =============================================================================
df_res = pd.DataFrame(results)
print("\n=== RESUMO ===")
cols = ["algo","n_clusters","n_noise","silhouette","davies_bouldin","calinski_harabasz",
        "intra_var","connectivity","ARI","jaccard"]
print(df_res[cols].round(4).to_string(index=False))

# Estabilidade por bootstrap (3 runs de 50% com K-Means++ best_k)
print(f"\n[Estabilidade K-Means++ k={best_k}] 3 bootstraps de 50%")
rng = np.random.default_rng(RNG)
ari_pairs = []
labels_full = []
for b in range(3):
    sub = rng.choice(len(X_famd_no), len(X_famd_no)//2, replace=False)
    km_b = KMeans(n_clusters=best_k, init="k-means++", n_init=10, random_state=RNG+b)
    lab_b = km_b.fit_predict(X_famd_no[sub])
    full = np.full(len(X_famd_no), -1)
    full[sub] = lab_b
    labels_full.append(full)
for i in range(len(labels_full)):
    for j in range(i+1, len(labels_full)):
        common = (labels_full[i] != -1) & (labels_full[j] != -1)
        ari_pairs.append(adjusted_rand_score(labels_full[i][common], labels_full[j][common]))
print(f"  ARI pareado entre bootstraps: {ari_pairs}")
print(f"  média = {np.mean(ari_pairs):.3f}  (>0.7 estável)")

print("\n[OK]")
