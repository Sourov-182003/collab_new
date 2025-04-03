from flask import Flask, request, jsonify, render_template
from waitress import serve
import pickle
from surprise import SVD
from flask_cors import CORS

app = Flask(__name__)

# Enhanced CORS configuration
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:5173", "https://instamart-ejm2.onrender.com"],
        "methods": ["GET", "POST", "OPTIONS", "PUT", "DELETE"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True,
        "max_age": 86400
    }
})

# Add this after CORS initialization to handle OPTIONS requests
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:5173')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

# Load saved models and data with error handling
print("Loading model and data...")
try:
    with open("Model_svd.pkl", "rb") as f:
        svd_algo = pickle.load(f)
    print("SVD Model loaded successfully.")

    with open("user_item.pkl", "rb") as f:
        user_item = pickle.load(f)
    print("User-Item Data loaded successfully.")

    with open("product_names.pkl", "rb") as f:
        product_names = pickle.load(f)
    print("Product Names loaded successfully.")

    with open("product_aisles.pkl", "rb") as f:
        product_aisles = pickle.load(f)
    print("Product Aisles loaded successfully.")
except Exception as e:
    print(f"Error loading files: {e}")
    exit(1)

@app.route("/")
def home():
    """Render the home page with the recommendation UI"""
    return render_template("index.html")

@app.route("/recommend/past", methods=["GET"])
def recommend_past():
    """Get past products for a user"""
    try:
        user_id = int(request.args.get("user_id", 1))

        if user_id not in user_item:
            return jsonify({"error": f"User ID {user_id} not found."}), 400

        # Get past products the user has interacted with
        past_products = user_item.get(user_id, {})
        result = [{"product": product_names[pid], "rating": round(rating, 2)}
                  for pid, rating in past_products.items() if pid in product_names]

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/recommend", methods=["GET"])
def recommend():
    """Get top recommended items for a user"""
    try:
        user_id = int(request.args.get("user_id", 1))
        n_recommendations = int(request.args.get("n", 10))

        if user_id not in user_item:
            return jsonify({"error": f"User ID {user_id} not found."}), 400

        interacted = set(user_item.get(user_id, {}))
        unseen = list(set(product_names.keys()) - interacted)

        if not unseen:
            return jsonify({"message": "No new products to recommend."})

        predictions = [(pid, svd_algo.predict(user_id, pid).est) for pid in unseen]
        top_n = sorted(predictions, key=lambda x: -x[1])[:n_recommendations]

        result = [{"product": product_names[pid], "rating": round(rating, 2)}
                  for pid, rating in top_n if pid in product_names]
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/recommend/aisle", methods=["GET"])
def recommend_aisle():
    """Get recommended items for a user in a specific aisle"""
    try:
        user_id = int(request.args.get("user_id", 1))
        aisle_name = request.args.get("aisle", "cookies cakes").strip().lower()
        n_recommendations = int(request.args.get("n", 10))

        aisle_products = {pid for pid, aisle in product_aisles.items() if aisle.lower() == aisle_name}
        if not aisle_products:
            return jsonify({"error": f"No products found in aisle '{aisle_name}'."}), 400

        if user_id not in user_item:
            return jsonify({"error": f"User ID {user_id} not found."}), 400

        interacted = set(user_item.get(user_id, {}))
        unseen = list(aisle_products - interacted)

        if not unseen:
            return jsonify({"message": "No new products to recommend in this aisle."})

        predictions = [(pid, svd_algo.predict(user_id, pid).est) for pid in unseen]
        top_n = sorted(predictions, key=lambda x: -x[1])[:n_recommendations]

        result = [{"product": product_names[pid], "aisle": aisle_name, "rating": round(rating, 2)}
                  for pid, rating in top_n if pid in product_names]
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    #app.run(host="0.0.0.0", port=5002, debug=True)
    serve(app,host="0.0.0.0",port=5003)
