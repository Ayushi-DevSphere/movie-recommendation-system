import os
from flask import Flask, jsonify, request, send_from_directory
from src.preprocess import preprocess_full
from src.recommender import build_model, recommend

app = Flask(__name__, static_folder="static", static_url_path="/static")

# Load data and build model at startup
print("[*] Loading and preprocessing data ...")
df = preprocess_full()
_, similarity_matrix = build_model(df)
print(f"[*] Model ready — {len(df)} movies loaded.")

ALL_GENRES = sorted({g for genres in df["genres_list"] for g in genres})


def movie_to_dict(row):
    """Convert a DataFrame row to a JSON-friendly dict."""
    year = str(row["release_date"])[:4] if row.get("release_date") else ""
    return {
        "id": int(row["id"]),
        "title": row["title"],
        "overview": row.get("overview", ""),
        "rating": round(float(row.get("vote_average", 0)), 1),
        "year": year,
        "genres": row.get("genres_list", []),
    }


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/genres")
def api_genres():
    return jsonify(ALL_GENRES)


@app.route("/api/movies")
def api_movies():
    return jsonify([movie_to_dict(df.iloc[i]) for i in range(len(df))])


@app.route("/api/movies/trending")
def api_trending():
    top = df.nlargest(20, "vote_average")
    return jsonify([movie_to_dict(top.iloc[i]) for i in range(len(top))])


@app.route("/api/movies/genre/<genre>")
def api_by_genre(genre):
    mask = df["genres_list"].apply(lambda gl: genre in gl)
    filtered = df[mask].nlargest(20, "vote_average")
    return jsonify([movie_to_dict(filtered.iloc[i]) for i in range(len(filtered))])


@app.route("/api/recommend/<title>")
def api_recommend(title):
    top_n = request.args.get("n", 10, type=int)
    try:
        results = recommend(title, df, similarity_matrix, top_n=top_n)
        recs = []
        for rec_title, score in results:
            row = df[df["title"] == rec_title].iloc[0]
            d = movie_to_dict(row)
            d["similarity"] = score
            recs.append(d)
        return jsonify({"movie": title, "recommendations": recs})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404


@app.route("/api/search")
def api_search():
    q = request.args.get("q", "").strip().lower()
    if not q:
        return jsonify([])
    matches = df[df["title"].str.lower().str.contains(q, na=False)].head(10)
    return jsonify([movie_to_dict(matches.iloc[i]) for i in range(len(matches))])


@app.route("/api/movie/<int:movie_id>")
def api_movie_detail(movie_id):
    match = df[df["id"] == movie_id]
    if match.empty:
        return jsonify({"error": "Movie not found"}), 404
    return jsonify(movie_to_dict(match.iloc[0]))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
