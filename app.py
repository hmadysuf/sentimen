import os
import json
import pandas as pd
from flask import (
    Flask, render_template, request, redirect,
    url_for, flash, jsonify, send_from_directory
)
from config import Config

from modules.crawler import crawl_from_url, crawl_mbg_data
from modules.preprocess import preprocess_dataframe
from modules.labeling import label_dataframe
from modules.modeling import train_models
from modules.evaluator import (
    predict_text, plot_sentiment_distribution, plot_confusion_matrix,
    plot_comment_vs_reply,
)

app = Flask(__name__)
app.config.from_object(Config)
app.secret_key = Config.SECRET_KEY


# ==================== HELPER ====================
def load_comments():
    if os.path.exists(Config.COMMENTS_CSV):
        return pd.read_csv(Config.COMMENTS_CSV)
    return pd.DataFrame()


def load_metrics():
    if os.path.exists(Config.METRICS_JSON):
        with open(Config.METRICS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# ==================== ROUTES ====================
@app.route("/")
def index():
    df = load_comments()
    metrics = load_metrics()
    total = len(df) if not df.empty else 0
    dist = {}
    if not df.empty and "label" in df.columns:
        dist = df["label"].value_counts().to_dict()

    # Generate chart distribusi
    chart_path = os.path.join(Config.STATIC_DIR, "charts", "distribusi.png")
    if not df.empty and "label" in df.columns:
        plot_sentiment_distribution(df, "label", chart_path)

    return render_template(
        "index.html",
        total=total,
        dist=dist,
        metrics=metrics,
        has_data=not df.empty,
    )


@app.route("/analyze-url", methods=["GET", "POST"])
def analyze_url():
    result = None
    if request.method == "POST":
        youtube_url = request.form.get("youtube_url", "").strip()
        try:
            if not Config.YOUTUBE_API_KEY:
                raise ValueError("YOUTUBE_API_KEY belum di-set.")

            max_pages = int(request.form.get("max_pages", "5"))
            if not 1 <= max_pages <= 20:
                raise ValueError("Maks. halaman komentar harus antara 1 dan 20.")

            videos, comments = crawl_from_url(
                api_key=Config.YOUTUBE_API_KEY,
                url=youtube_url,
                max_pages=max_pages,
                include_replies=request.form.get("include_replies") == "on",
            )
            if comments.empty:
                analyzed = comments
                flash("Video ditemukan, tetapi tidak ada komentar yang dapat dianalisis.", "warning")
            else:
                analyzed = preprocess_dataframe(comments, text_col="comment_text")
                analyzed = label_dataframe(analyzed, tokens_col="tokens")

            dist_all = analyzed["label"].value_counts().to_dict() if "label" in analyzed else {}
            comment_rows = analyzed[analyzed["type"] == "comment"] if "type" in analyzed else analyzed
            reply_rows = analyzed[analyzed["type"] == "reply"] if "type" in analyzed else analyzed.iloc[0:0]

            if not analyzed.empty:
                charts_dir = os.path.join(Config.STATIC_DIR, "charts")
                plot_sentiment_distribution(
                    analyzed,
                    "label",
                    os.path.join(charts_dir, "distribusi_url.png"),
                )
                plot_comment_vs_reply(analyzed, Config.STATIC_DIR)

            result = {
                "url": youtube_url,
                "video": videos.iloc[0].to_dict(),
                "total": len(analyzed),
                "total_comment": len(comment_rows),
                "total_reply": len(reply_rows),
                "dist_all": dist_all,
                "dist_comment": comment_rows["label"].value_counts().to_dict() if "label" in comment_rows else {},
                "dist_reply": reply_rows["label"].value_counts().to_dict() if "label" in reply_rows else {},
                "samples": analyzed.head(100).to_dict(orient="records"),
            }
            if not comments.empty and analyzed.empty:
                flash("Komentar ditemukan, tetapi tidak ada teks yang dapat dianalisis.", "warning")
            elif not analyzed.empty:
                flash(f"Berhasil menganalisis {len(analyzed)} komentar dan reply.", "success")
        except (TypeError, ValueError) as e:
            flash(str(e), "warning")
        except Exception as e:
            flash(f"Error saat analisis URL: {e}", "danger")

    return render_template("analyze_url.html", result=result)


@app.route("/crawl", methods=["GET", "POST"])
def crawl():
    if request.method == "POST":
        try:
            df_v, df_c = crawl_mbg_data(
                api_key=Config.YOUTUBE_API_KEY,
                queries=Config.SEARCH_QUERIES,
                max_videos=Config.MAX_VIDEOS_PER_QUERY,
                max_pages=Config.MAX_COMMENT_PAGES_PER_VIDEO,
                videos_csv=Config.VIDEOS_CSV,
                comments_csv=Config.COMMENTS_CSV,
            )
            flash(f"Berhasil crawl {len(df_c)} komentar dari {len(df_v)} video.", "success")
        except Exception as e:
            flash(f"Error saat crawling: {e}", "danger")
        return redirect(url_for("crawl"))

    df = load_comments()
    preview = df.head(20).to_dict(orient="records") if not df.empty else []
    return render_template("crawl.html", preview=preview, total=len(df))


@app.route("/preprocessing", methods=["GET", "POST"])
def preprocessing():
    df = load_comments()
    if df.empty:
        flash("Belum ada data. Silakan crawl terlebih dahulu.", "warning")
        return redirect(url_for("crawl"))

    if request.method == "POST":
        try:
            df_proc = preprocess_dataframe(df, text_col="comment_text")
            df_proc = label_dataframe(df_proc, tokens_col="tokens")
            df_proc.to_csv(Config.COMMENTS_CSV, index=False, encoding="utf-8-sig")
            flash(f"Preprocessing & labeling selesai: {len(df_proc)} komentar.", "success")
        except Exception as e:
            flash(f"Error preprocessing: {e}", "danger")
        return redirect(url_for("preprocessing"))

    # Tampilkan preview
    has_proc = "stemmed" in df.columns
    preview = df.head(20).to_dict(orient="records")
    return render_template(
        "preprocessing.html",
        preview=preview,
        total=len(df),
        has_proc=has_proc,
    )


@app.route("/training", methods=["GET", "POST"])
def training():
    df = load_comments()
    if df.empty or "label" not in df.columns:
        flash("Data belum diproses/dilabeli. Lakukan preprocessing dulu.", "warning")
        return redirect(url_for("preprocessing"))

    if request.method == "POST":
        try:
            results = train_models(
                df=df,
                text_col="stemmed",
                label_col="label",
                model_dir=Config.MODEL_DIR,
                metrics_path=Config.METRICS_JSON,
            )
            flash("Training selesai: Naive Bayes & SVM berhasil dilatih.", "success")
            return redirect(url_for("evaluation"))
        except Exception as e:
            flash(f"Error training: {e}", "danger")
            return redirect(url_for("training"))

    metrics = load_metrics()
    return render_template("training.html", metrics=metrics)


@app.route("/evaluation")
def evaluation():
    metrics = load_metrics()
    if not metrics:
        flash("Belum ada hasil evaluasi. Latih model dulu.", "warning")
        return redirect(url_for("training"))

    # Buat confusion matrix chart
    for name in ["naive_bayes", "svm"]:
        if name in metrics:
            cm = metrics[name]["confusion_matrix"]
            labels = metrics[name]["labels"]
            out = os.path.join(Config.STATIC_DIR, "charts", f"cm_{name}.png")
            plot_confusion_matrix(cm, labels, f"Confusion Matrix - {name.upper()}", out)

    return render_template("evaluation.html", metrics=metrics)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    if request.method == "POST":
        text = request.form.get("komentar", "").strip()
        if not text:
            flash("Masukkan komentar terlebih dahulu.", "warning")
        else:
            try:
                result = predict_text(text, Config.MODEL_DIR)
            except Exception as e:
                flash(f"Error prediksi: {e}", "danger")
    return render_template("predict.html", result=result)


@app.route("/api/metrics")
def api_metrics():
    return jsonify(load_metrics() or {})


@app.route("/api/distribution")
def api_distribution():
    df = load_comments()
    if df.empty or "label" not in df.columns:
        return jsonify({})
    return jsonify(df["label"].value_counts().to_dict())


@app.errorhandler(404)
def not_found(e):
    return render_template("base.html", error="Halaman tidak ditemukan"), 404


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)