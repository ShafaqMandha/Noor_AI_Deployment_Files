from flask import Flask, request, render_template_string, send_from_directory, jsonify, send_file, render_template
from flask_cors import CORS
import pandas as pd
import os
from openai import OpenAI
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app with dotenv disabled
app = Flask(__name__, static_url_path='')
app.config['ENV'] = 'development'
app.config['DEBUG'] = True
CORS(app)  # Enable CORS for all routes

# Define skin types
skin_cols = ['Dry', 'Normal', 'Oily']

# Initialize OpenAI client with API key from environment variable
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

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
        return send_file('model.json')
    except Exception as e:
        return f"Error serving model.json: {str(e)}", 500

@app.route('/metadata.json')
def serve_metadata():
    try:
        return send_file('metadata.json')
    except Exception as e:
        return f"Error serving metadata.json: {str(e)}", 500

@app.route('/weights.bin')
def serve_weights():
    try:
        return send_file('weights.bin')
    except Exception as e:
        return f"Error serving weights.bin: {str(e)}", 500

# API endpoint for recommendations
@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    data = request.get_json()
    skin_type = data.get('skin_type', '').strip()
    
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
        budget = data.get('budget')
        recommendations = recommend_products(skin_type, budget)
        
        # Convert recommendations to dict and add a description
        recs_dict = recommendations.to_dict('records')
        for rec in recs_dict:
            rec['description'] = f"{rec['Label']} for {skin_type} skin"
            
        return {'recommendations': recs_dict}
    except Exception as e:
        return {'error': str(e)}, 500

def get_skincare_context(df):
    """Create a context string from the skincare data"""
    if df is None or df.empty:
        return "No data available."
        
    try:
        context = """You are a skincare assistant that MUST ONLY provide recommendations and information based on the following dataset. 
        DO NOT make recommendations about products that are not in this dataset.
        DO NOT provide general skincare advice without referencing specific products from the dataset.
        
        Dataset Overview:
        """
        
        # Add general statistics
        context += f"Total number of products: {len(df)}\n"
        context += f"Product categories: {', '.join(df['Label'].dropna().unique())}\n"
        context += f"Brands available: {', '.join(df['brand'].dropna().unique())}\n\n"
        
        # Add detailed product information (limit to first 10 products to avoid context length issues)
        context += "Available Products (showing first 10):\n"
        for _, row in df.head(10).iterrows():
            try:
                context += f"Product: {row['brand']} {row['name']}\n"
                context += f"Category: {row['Label']}\n"
                context += f"Price: ${row['price']}\n"
                context += f"Rating: {row['rank']}\n"
                context += f"Ingredients: {row['ingredients']}\n"
                context += "Suitable for skin types: "
                skin_types = []
                if row.get('Combination'): skin_types.append('Combination')
                if row.get('Dry'): skin_types.append('Dry')
                if row.get('Normal'): skin_types.append('Normal')
                if row.get('Oily'): skin_types.append('Oily')
                if row.get('Sensitive'): skin_types.append('Sensitive')
                context += ', '.join(skin_types) + "\n"
                context += f"Availability: {row.get('availability', 'Unknown')}\n\n"
            except Exception as e:
                continue
        
        return context
    except Exception as e:
        print(f"Error generating context: {str(e)}")
        return "Error generating context."

def get_chatbot_response(user_input, context, df):
    """Get response from OpenAI API"""
    if df is None or df.empty:
        return "I apologize, but I'm currently unable to access the product database. Please try again later."
        
    try:
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
            {"role": "user", "content": user_input}
        ]
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=800
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"I apologize, but I encountered an error: {str(e)}"

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')
        
        # Generate context from the dataset
        context = get_skincare_context(df)
        
        # Get response from OpenAI
        response = get_chatbot_response(user_message, context, df)
        
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Main route
@app.route('/')
def index():
    try:
        # Get dataset statistics
        stats = get_dataset_stats(df)
        
        # Read the HTML template
        with open('AI Beauty Advisor.html', 'r', encoding='utf-8') as file:
            html_content = file.read()
            
        # Add chatbot widget HTML and JavaScript
        chatbot_html = """
        <div id="chatbot-widget" style="position: fixed; bottom: 20px; right: 20px; z-index: 1000;">
            <div id="chatbot-header" style="background: #4a90e2; color: white; padding: 10px; cursor: pointer; border-radius: 5px 5px 0 0;">
                Chat with Noor
            </div>
            <div id="chatbot-body" style="display: none; background: white; border: 1px solid #ccc; height: 400px; width: 300px; border-radius: 0 0 5px 5px;">
                <div id="chat-messages" style="height: 320px; overflow-y: auto; padding: 10px;"></div>
                <div style="padding: 10px; border-top: 1px solid #ccc;">
                    <input type="text" id="chat-input" style="width: 80%; padding: 5px;" placeholder="Type your message...">
                    <button onclick="sendMessage()" style="width: 18%; padding: 5px;">Send</button>
                </div>
            </div>
        </div>

        <script>
            document.getElementById('chatbot-header').addEventListener('click', function() {
                const body = document.getElementById('chatbot-body');
                body.style.display = body.style.display === 'none' ? 'block' : 'none';
            });

            function sendMessage() {
                const input = document.getElementById('chat-input');
                const message = input.value.trim();
                if (!message) return;

                // Add user message to chat
                addMessage('You: ' + message, 'user');
                input.value = '';

                // Send to backend
                fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ message: message })
                })
                .then(response => response.json())
                .then(data => {
                    addMessage('Noor: ' + data.response, 'bot');
                })
                .catch(error => {
                    addMessage('Error: Could not get response', 'error');
                });
            }

            function addMessage(message, type) {
                const messagesDiv = document.getElementById('chat-messages');
                const messageElement = document.createElement('div');
                messageElement.style.marginBottom = '10px';
                messageElement.style.padding = '5px';
                messageElement.style.borderRadius = '5px';
                
                if (type === 'user') {
                    messageElement.style.backgroundColor = '#e3f2fd';
                    messageElement.style.marginLeft = '20%';
                } else if (type === 'bot') {
                    messageElement.style.backgroundColor = '#f5f5f5';
                    messageElement.style.marginRight = '20%';
                } else {
                    messageElement.style.backgroundColor = '#ffebee';
                }
                
                messageElement.textContent = message;
                messagesDiv.appendChild(messageElement);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }

            // Add enter key support
            document.getElementById('chat-input').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    sendMessage();
                }
            });
        </script>
        """
        
        # Insert chatbot HTML before the closing body tag
        html_content = html_content.replace('</body>', chatbot_html + '</body>')
        
        return html_content
    except Exception as e:
        print(f"Error serving main page: {str(e)}")
        return jsonify({"error": "Main page not found"}), 404

if __name__ == '__main__':
    print("\nStarting Flask server...")
    app.run(debug=True, port=5000, load_dotenv=False)
    print("Server will be available at http://localhost:5000")
