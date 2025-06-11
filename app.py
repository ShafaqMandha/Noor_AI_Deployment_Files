from flask import Flask, request, render_template_string, send_from_directory, jsonify, send_file, render_template
from flask_cors import CORS
import pandas as pd
import os
from openai import OpenAI
import json

# Initialize Flask app with dotenv disabled
app = Flask(__name__, static_url_path='')
app.config['ENV'] = 'development'
app.config['DEBUG'] = True
CORS(app)  # Enable CORS for all routes

# Define skin types
skin_cols = ['Dry', 'Normal', 'Oily']

# Initialize OpenAI client with direct API key
try:
    OPENAI_API_KEY = "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"
    client = OpenAI(api_key=OPENAI_API_KEY)
    print("OpenAI client initialized successfully!")
except Exception as e:
    print(f"Error initializing OpenAI client: {str(e)}")
    raise

# Load and clean the dataset
try:
    print("Loading dataset...")
    df = pd.read_csv('cosmetic_p.csv')
    print("Dataset loaded successfully!")
    
    # Clean the dataset
    df = df.dropna(subset=['name', 'brand', 'price', 'rank', 'ingredients'])
    df['ingredients'] = df['ingredients'].str.lower()
    print("Dataset cleaning completed!")
except Exception as e:
    print(f"Error loading dataset: {str(e)}")
    raise

def get_dataset_stats(df):
    """Get statistics about the dataset"""
    if df is None or df.empty:
        return {}
    
    try:
        stats = {
            'total_products': len(df),
            'brands': df['brand'].nunique() if 'brand' in df.columns else 0,
            'categories': df['Label'].nunique() if 'Label' in df.columns else 0,
            'avg_rating': round(df['rank'].mean(), 1) if 'rank' in df.columns else 0
        }
        return stats
    except Exception:
        return {}

# Recommendation logic
def recommend_products(skin_type, budget=None, top_n_per_type=2):
    # Products that are suitable for the specific skin type
    skin_type_products = df[df[skin_type] == 1]
    
    # Apply budget filter if specified
    if budget:
        skin_type_products = skin_type_products[skin_type_products["price"] <= budget]
    
    # Sort by category and rank
    filtered_sorted = skin_type_products.sort_values(by=["Label", "rank"], ascending=[True, False])
    
    # Get top N products per category
    recommendations = filtered_sorted.groupby("Label").head(top_n_per_type)
    
    return recommendations[["Label", "brand", "name", "price", "rank", "Available at"]]

# Serve model files
@app.route('/model.json')
def serve_model():
    try:
        return send_file(os.path.abspath('model.json'))
    except Exception as e:
        print(f"Error serving model.json: {str(e)}")
        return f"Error serving model.json: {str(e)}", 500

@app.route('/metadata.json')
def serve_metadata():
    try:
        return send_file(os.path.abspath('metadata.json'))
    except Exception as e:
        print(f"Error serving metadata.json: {str(e)}")
        return f"Error serving metadata.json: {str(e)}", 500

@app.route('/weights.bin')
def serve_weights():
    try:
        return send_file(os.path.abspath('weights.bin'))
    except Exception as e:
        print(f"Error serving weights.bin: {str(e)}")
        return f"Error serving weights.bin: {str(e)}", 500

# API endpoint for recommendations
@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    data = request.get_json()
    skin_type = data.get('skin_type', '').strip()
    budget = data.get('budget')
    
    # Normalize skin type
    skin_type_map = {
        'dry skin': 'Dry',
        'normal skin': 'Normal',
        'oily skin': 'Oily',
        'dry': 'Dry',
        'normal': 'Normal',
        'oily': 'Oily'
    }
    skin_type = skin_type_map.get(skin_type.lower(), skin_type)
    
    if skin_type not in skin_cols:
        return {'error': f'Invalid skin type: {skin_type}. Must be one of: {", ".join(skin_cols)}'}, 400
        
    try:
        recommendations = recommend_products(skin_type, budget)
        
        # Convert recommendations to dict and add a description
        recs_dict = recommendations.to_dict('records')
        for rec in recs_dict:
            rec['description'] = f"{rec['Label']} for {skin_type} skin"
            
        return {'recommendations': recs_dict}
    except Exception as e:
        return {'error': str(e)}, 500

def get_skincare_context(df, recommended_products=None):
    """Create a context string from the skincare data"""
    if df is None or df.empty:
        print("DEBUG: Dataset is empty or None in get_skincare_context.")
        return "No data available."
        
    try:
        context = """You are Noor, a skincare assistant that provides personalized skincare recommendations.
        You MUST ONLY provide recommendations and information based on the following dataset.
        DO NOT make recommendations about products that are not in this dataset.
        DO NOT provide general skincare advice without referencing specific products from the dataset.
        
        Dataset Overview:
        """
        
        # Add general statistics
        context += f"Total number of products: {len(df)}\n"
        context += f"Product categories: {', '.join(df['Label'].dropna().unique())}\n"
        context += f"Brands available: {', '.join(df['brand'].dropna().unique())}\n\n"
        
        # Add recommended products if provided
        if recommended_products:
            print(f"DEBUG: Recommended products received: {recommended_products}")
            context += "Currently Recommended Products:\n"
            for product in recommended_products:
                context += f"Product: {product['brand']} {product['name']}\n"
                context += f"Category: {product['Label']}\n"
                context += f"Price: ${product['price']}\n"
                context += f"Rating: {product['rank']}\n"
                context += f"Description: {product.get('description', '')}\n\n"
        else:
            print("DEBUG: No recommended products provided to get_skincare_context.")
        
        print(f"DEBUG: Generated context length: {len(context)}")
        return context
    except Exception as e:
        print(f"Error generating context: {str(e)}")
        return "Error generating context."

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        recommended_products = data.get('recommended_products', [])
        
        print(f"DEBUG: Received chat message: '{user_message}'")
        print(f"DEBUG: Received recommended products in chat endpoint: {recommended_products}")
        
        if not user_message:
            print("DEBUG: No message provided.")
            return jsonify({'error': 'No message provided'}), 400
            
        # Get context with recommended products
        context = get_skincare_context(df, recommended_products)
        print(f"DEBUG: Context for OpenAI: {context[:200]}...") # Print first 200 chars of context
        
        # Prepare messages for OpenAI
        messages = [
            {"role": "system", "content": f"""You are Noor, a skincare assistant that provides personalized skincare recommendations.
            
{context}

STRICT GUIDELINES:
1. ONLY recommend products that are available in our collection
2. When suggesting products, use this conversational format:
   "I can recommend [product name] by [brand] for [price] - Available at: [availability information]
   This [product type] contains ingredients like [key ingredients], which is beneficial for [skin type]:
   * Primary benefit: [main benefit]
   * Secondary benefits: [additional benefits]"

3. For ingredient questions:
   - Provide scientifically-backed benefits of ingredients
   - Include specific examples of products that contain these ingredients
   - Format ingredient benefits as follows:
     * Primary benefit: [main benefit]
     * Secondary benefits: [additional benefits]
     * How it works: [brief scientific explanation]
     * Products containing this ingredient: [list relevant products]
   - Only discuss ingredients that are present in our products
   - Be precise and accurate about ingredient benefits
   - Avoid making unsubstantiated claims
   - If an ingredient's benefits are not well-documented, acknowledge this limitation

4. For routine building:
   - Only suggest products from our collection
   - Consider skin type compatibility
   - Consider product categories (cleanser, moisturizer, etc.)
   - Consider price points
   - Use the same format as above for each product

5. If a user asks about products or ingredients we don't carry:
   - Politely explain that you can only provide information about products we offer
   - Suggest similar products that are available using the format above

6. Keep responses focused on the available products
7. Always verify product availability before recommending

Remember: You can ONLY make recommendations based on the products we offer."""},
            {"role": "user", "content": user_message}
        ]
        
        print(f"DEBUG: Messages sent to OpenAI: {messages}")
        
        # Get response from OpenAI
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=800
        )
        
        # Extract and return the response
        bot_response = response.choices[0].message.content
        print(f"DEBUG: OpenAI raw response: {response}")
        print(f"DEBUG: Extracted bot response: '{bot_response}'")
        return jsonify({'response': bot_response})
        
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': 'An error occurred while processing your request. Please try again.'}), 500

@app.route('/')
def landing():
    try:
        with open('landing_page.html', 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        return f"Error loading landing page: {str(e)}", 500

@app.route('/main')
def main():
    try:
        with open('AI Beauty Advisor.html', 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        return f"Error loading main page: {str(e)}", 500

@app.route('/thank_you')
def thank_you():
    try:
        with open('thank_you.html', 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        return f"Error loading thank you page: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)