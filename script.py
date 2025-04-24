from flask import Flask, render_template, request, jsonify, redirect, session, url_for
import pandas as pd
import datetime
import os
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = "super-secret-key"

# 感情分析モデル
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    framework="pt"
)

# 要約モデル
summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    framework="pt"
)

# ポイントからレベル判定
def get_level(points):
    if points < 10:
        return 0
    elif points < 30:
        return 1
    elif points < 60:
        return 2
    else:
        return 3

# ポイント追加＋レベルアップチェック
def add_points_and_check_levelup(user, current_date, points_to_add, progress_file="avatar_progress.csv"):
    ym = current_date.strftime("%Y-%m")

    if os.path.exists(progress_file):
        df = pd.read_csv(progress_file)
    else:
        df = pd.DataFrame(columns=["user", "year_month", "points", "level", "char_type"])

    mask = (df["user"] == user) & (df["year_month"] == ym)

    if mask.any():
        old_points = int(df.loc[mask, "points"].values[0])
        old_level = get_level(old_points)
        new_points = old_points + points_to_add
        new_level = get_level(new_points)
        df.loc[mask, "points"] = new_points
        df.loc[mask, "level"] = new_level
        level_up = new_level > old_level
    else:
        new_points = points_to_add
        new_level = get_level(new_points)
        df = pd.concat([df, pd.DataFrame([[user, ym, new_points, new_level, 1]], columns=df.columns)], ignore_index=True)
        level_up = new_level > 0

    df.to_csv(progress_file, index=False)
    return level_up

# 最初にログインページにリダイレクト
@app.route("/")
def root():
    return redirect(url_for("login_form"))

# ログインページ
@app.route("/login", methods=["GET", "POST"])
def login_form():
    if request.method == "POST":
        username = request.form.get("username")
        if username:
            session["user"] = username
            return redirect(url_for("home"))
    return render_template("login.html")

# ログアウト
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login_form"))

# ホーム（カレンダー）
@app.route("/home", methods=["GET"])
def home():
    user = session.get("user")
    if not user:
        return redirect("/login")

    level_up = request.args.get("level_up", "0") == "1"

    today = datetime.date.today()
    ym = today.strftime("%Y-%m")
    progress_file = "avatar_progress.csv"

    def get_remaining_points(points):
        if points < 10:
            return 10 - points
        elif points < 30:
            return 30 - points
        elif points < 60:
            return 60 - points
        else:
            return 0

    if os.path.exists(progress_file):
        df = pd.read_csv(progress_file)
    else:
        df = pd.DataFrame(columns=["user", "year_month", "points", "level", "char_type"])

    user_row = df[(df["user"] == user) & (df["year_month"] == ym)]

    if user_row.empty:
        points = 0
        level = 0
        char_type = 1
        remaining = 10
        new_row = pd.DataFrame([[user, ym, points, level, char_type]], columns=df.columns)
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(progress_file, index=False)
    else:
        points = int(user_row.iloc[0]["points"])
        char_type = int(user_row.iloc[0]["char_type"])
        level = get_level(points)
        remaining = get_remaining_points(points)

    return render_template("index.html",
                           user=user,
                           level=level,
                           char_type=char_type,
                           remaining=remaining,
                           points=points,
                           level_up=level_up)

# 日記入力ページ
@app.route("/write")
def write():
    user = session.get("user")
    if not user:
        return redirect(url_for("login_form"))
    return render_template("write.html", user=user)

# 日記投稿処理
@app.route("/submit", methods=["POST"])
def post_diary():
    user = session.get("user")
    if not user:
        return redirect(url_for("login_form"))

    diary_text = request.form["diary"]
    today = datetime.date.today().isoformat()

    sentiment_results = sentiment_analyzer(diary_text)
    star_score = int(sentiment_results[0]['label'][0])
    sentiment_score = sentiment_results[0]['score']
    sentiment = "ポジティブ" if star_score >= 4 else "ニュートラル" if star_score == 3 else "ネガティブ"

    point_gain = int(sentiment_score * 10) if sentiment == "ポジティブ" else 0
    level_up = add_points_and_check_levelup(user, datetime.date.today(), point_gain)

    if len(diary_text) > 100:
        summary = summarizer(diary_text, max_length=60, min_length=20, do_sample=False)[0]['summary_text']
    else:
        summary = diary_text

    font_path = "fonts/NotoSansJP-VariableFont_wght.ttf"
    wordcloud = WordCloud(width=800, height=400, background_color='white', font_path=font_path).generate(diary_text)
    img_path = f"static/wordcloud_{today}_{user}.png"
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(img_path)
    plt.close()

    log_file = "daily_highlight_log.csv"
    if os.path.exists(log_file):
        df = pd.read_csv(log_file)
    else:
        df = pd.DataFrame(columns=["date", "summary", "sentiment", "score", "text", "user"])

    new_entry = pd.DataFrame([[today, summary, sentiment, sentiment_score, diary_text, user]],
                             columns=df.columns)
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(log_file, index=False)

    return redirect(url_for("home", level_up="1" if level_up else "0"))

# カレンダーデータ取得
@app.route("/calendar")
def calendar_data():
    user = session.get("user")
    if not user or not os.path.exists("daily_highlight_log.csv"):
        return jsonify([])

    df = pd.read_csv("daily_highlight_log.csv")
    df = df[df["user"] == user]

    events = []
    for _, row in df.iterrows():
        color = "#00e676" if row["sentiment"] == "ポジティブ" else "#ffca28" if row["sentiment"] == "ニュートラル" else "#ff1744"
        events.append({
            "title": row["summary"],
            "start": row["date"],
            "color": color,
            "sentiment": row["sentiment"],
            "score": float(row["score"]),
            "text": row["text"]
        })

    return jsonify(events)

# グラフページ（感情推移）
@app.route("/graph")
def emotion_graph():
    user = session.get("user")
    if not user or not os.path.exists("daily_highlight_log.csv"):
        return render_template("graph.html", graph_data=[])

    df = pd.read_csv("daily_highlight_log.csv")
    today = datetime.date.today()
    ym = today.strftime("%Y-%m")

    # ✅ ここから
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["user"] == user) & (df["date"].dt.strftime("%Y-%m") == ym)]

    # 💡 ここを置き換える！
    df["mapped"] = df["sentiment"].map({
        "ポジティブ": 1,
        "ニュートラル": 0,
        "ネガティブ": -1
    })
    df_grouped = df.groupby(df["date"].dt.strftime("%Y-%m-%d"))["mapped"].sum().reset_index()
    graph_data = [{"date": row["date"], "value": row["mapped"]} for _, row in df_grouped.iterrows()]
    # ✅ ここまでで graph_data が完成

    return render_template("graph.html", graph_data=graph_data)


if __name__ == "__main__":
    app.run(debug=True)
