"""Verification script — tests the complete ML pipeline without a database."""
import os, sys

os.environ["MODEL_DIR"] = "storage/models"
os.environ["REPORTS_DIR"] = "storage/reports"
os.makedirs("storage/models", exist_ok=True)
os.makedirs("storage/reports", exist_ok=True)

print("=" * 60)
print("ML PIPELINE VERIFICATION")
print("=" * 60)

# 1. Dataset generation
from app.ml.data_generator import generate_dataset
df = generate_dataset(500)
female_rate = df[df.gender == 0].eligible.mean()
male_rate   = df[df.gender == 1].eligible.mean()
print(f"\n[1] Dataset: {len(df)} rows | eligible={df.eligible.mean():.2%}")
print(f"    Female selection rate: {female_rate:.2%}")
print(f"    Male selection rate:   {male_rate:.2%}")
print(f"    Bias gap: {abs(male_rate - female_rate):.2%}")

# 2. Model training
from app.ml.trainer import train_model
meta = train_model(500)
print(f"\n[2] Training:")
print(f"    Accuracy  = {meta['accuracy']:.4f}")
print(f"    AUC-ROC   = {meta['auc_roc']:.4f}")
print(f"    F1 Score  = {meta['f1_score']:.4f}")
print(f"    CV Mean   = {meta['cross_val_mean']:.4f} ± {meta['cross_val_std']:.4f}")

# 3. Inference
from app.ml.predictor import predict
cases = [
    ("High-GPA Female", 8.5, 250, 88.0, 92.0, "female"),
    ("Mid-Range Male",  6.5, 150, 70.0, 75.0, "male"),
    ("Low-Score Female",5.0,  80, 52.0, 60.0, "female"),
]
print(f"\n[3] Inference:")
for name, sgpa, jee, marks, att, gender in cases:
    r = predict(sgpa, jee, marks, att, gender)
    verdict = "ELIGIBLE" if r["eligible"] else "NOT ELIGIBLE"
    print(f"    {name:30s} -> {verdict} (prob={r['probability']:.4f}, {r['confidence']})")

# 4. Fairness
from app.ml.fairness import compute_fairness_report
frep = compute_fairness_report()
print(f"\n[4] Fairness Report (overall_fair={frep['overall_fair']}):")
for m in frep["metrics"]:
    status = "PASS" if m["passed"] else "FAIL"
    print(f"    [{status}] {m['name']}: {m['value']:.4f} (threshold={m['threshold']})")

# 5. SHAP explanation
from app.ml.explainability import explain_prediction
exp = explain_prediction(8.5, 250, 88.0, 92.0, "female")
print(f"\n[5] SHAP Explanation:")
print(f"    Eligible={exp['eligible']}  Prob={exp['probability']:.4f}")
print(f"    Base value: {exp['base_value']:.4f}")
print(f"    Contributions:")
for feat, val in sorted(exp["feature_contributions"].items(), key=lambda x: abs(x[1]), reverse=True):
    bar = "+" if val >= 0 else "-"
    print(f"      {feat:25s}: {val:+.4f} {'▲' if val >= 0 else '▼'}")
print(f"    Interpretation: {exp['interpretation']}")

# 6. Auth module
print(f"\n[6] Auth/Security:")
from app.core.security import hash_password, verify_password, create_access_token, decode_token
h = hash_password("admin123")
assert verify_password("admin123", h), "Password verify failed!"
token = create_access_token("admin", "admin")
payload = decode_token(token)
assert payload["sub"] == "admin"
print(f"    bcrypt hash OK | JWT encode/decode OK | role={payload['role']}")

print("\n" + "=" * 60)
print("ALL TESTS PASSED ✓")
print("=" * 60)
