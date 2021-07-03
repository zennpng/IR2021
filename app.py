from flask import Flask, render_template, request
import engine

app = Flask(__name__, static_url_path='/static')

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/recommendations", methods=['POST'])
def recommendations():
    query = request.form.get("query")
    number = request.form.get("number")
    recommendations = engine.generate_recommendations(query, number)
    #recommendations = ["mr. vegas - party tun up, 2012 reggae","citizen king - better days, 1999 pop","fall out boy - champion, 2018 rock"]
    for song in range(len(recommendations)): 
        recommendations[song] = str(song+1) + ". " + recommendations[song]
    return render_template('recommendations.html', recommendations = recommendations)

 
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)