import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import make_pipeline

from data import generate_transactions
from features import CATEGORICAL, NUMERIC, add_features, build_preprocessor

REPORTS = Path(__file__).resolve().parent.parent / "reports"
REVIEW_COST = 5.0


def split_by_time(df, test_frac=0.25):
    cut = int(len(df) * (1 - test_frac))
    return df.iloc[:cut], df.iloc[cut:]


def business_cost(y_true, amount, flagged):
    missed = amount[(y_true == 1) & (~flagged)].sum()
    reviews = flagged.sum() * REVIEW_COST
    return missed + reviews


def best_threshold(y_true, proba, amount):
    grid = np.linspace(0.005, 0.95, 190)
    costs = np.array([business_cost(y_true, amount, proba >= t) for t in grid])
    return grid, costs, grid[costs.argmin()]


def main():
    REPORTS.mkdir(exist_ok=True)
    df = add_features(generate_transactions())
    train_full, test = split_by_time(df)
    train, val = split_by_time(train_full, 0.2)
    feats = NUMERIC + CATEGORICAL
    X_tr, y_tr = train[feats], train["is_fraud"]
    X_val, y_val, amount_val = val[feats], val["is_fraud"].to_numpy(), val["amount"].to_numpy()
    X_te, y_te = test[feats], test["is_fraud"]
    amount_te = test["amount"].to_numpy()

    models = {
        "Regresión logística": make_pipeline(
            build_preprocessor(True), LogisticRegression(max_iter=1000, class_weight="balanced")),
        "Gradient Boosting": make_pipeline(
            build_preprocessor(False),
            HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=24,
                                           l2_regularization=1.0, random_state=0)),
    }

    results, probas, val_scores = {}, {}, {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        val_scores[name] = average_precision_score(y_val, model.predict_proba(X_val)[:, 1])
        p = model.predict_proba(X_te)[:, 1]
        probas[name] = p
        results[name] = {"pr_auc": round(average_precision_score(y_te, p), 4),
                         "roc_auc": round(roc_auc_score(y_te, p), 4)}

    champion = max(val_scores, key=val_scores.get)
    p = probas[champion]
    p_val = models[champion].predict_proba(X_val)[:, 1]
    grid, costs_val, thr = best_threshold(y_val, p_val, amount_val)
    costs = np.array([business_cost(y_te.to_numpy(), amount_te, p >= t) for t in grid])
    flagged = p >= thr
    tp = int(((y_te == 1) & flagged).sum())
    fn = int(((y_te == 1) & ~flagged).sum())
    fp = int(((y_te == 0) & flagged).sum())

    no_model_cost = float(amount_te[y_te.to_numpy() == 1].sum())
    with_model_cost = float(business_cost(y_te.to_numpy(), amount_te, flagged))
    summary = {
        "fraud_rate_test": round(float(y_te.mean()), 5),
        "models": results,
        "champion": champion,
        "threshold": round(float(thr), 3),
        "precision": round(tp / max(tp + fp, 1), 3),
        "recall": round(tp / max(tp + fn, 1), 3),
        "cost_without_model": round(no_model_cost, 2),
        "cost_with_model": round(with_model_cost, 2),
        "savings_pct": round(100 * (1 - with_model_cost / no_model_cost), 1),
    }
    (REPORTS / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, pr in probas.items():
        prec, rec, _ = precision_recall_curve(y_te, pr)
        ax.plot(rec, prec, label=f"{name} (AP={results[name]['pr_auc']:.3f})", lw=2)
    ax.axhline(y_te.mean(), color="gray", ls="--", label="Azar")
    ax.set(xlabel="Recall", ylabel="Precision", title="Curva Precision-Recall")
    ax.legend(); fig.tight_layout(); fig.savefig(REPORTS / "pr_curve.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(grid, costs / 1000, lw=2)
    ax.axvline(thr, color="crimson", ls="--", label=f"Umbral elegido en validación = {thr:.2f}")
    ax.set(xlabel="Umbral de decisión", ylabel="Costo total (miles USD)", title="Costo de negocio según umbral")
    ax.legend(); fig.tight_layout(); fig.savefig(REPORTS / "cost_curve.png", dpi=150); plt.close(fig)

    sample = X_te.sample(20_000, random_state=0)
    imp = permutation_importance(models[champion], sample, y_te.loc[sample.index],
                                 scoring="average_precision", n_repeats=3, random_state=0, n_jobs=-1)
    order = imp.importances_mean.argsort()
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.barh(np.array(feats)[order], imp.importances_mean[order], color="#4c78a8")
    ax.set(xlabel="Caída de PR-AUC al permutar", title="Importancia de variables")
    fig.tight_layout(); fig.savefig(REPORTS / "feature_importance.png", dpi=150); plt.close(fig)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
