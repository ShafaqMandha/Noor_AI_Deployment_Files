# Noor AI Beauty Advisor

An AI-powered beauty advisor that provides personalized skincare recommendations based on skin type analysis.

## Features

- Skin type analysis using AI
- Personalized product recommendations
- Interactive chat interface
- Product information and ingredient details

## Setup Instructions

1. Clone the repository:
```bash
git clone <your-repository-url>
cd <repository-name>
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Add your OpenAI API key to the `.env` file:
     ```
     OPENAI_API_KEY=your_openai_api_key_here
     ```

5. Run the application:
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Environment Variables

Create a `.env` file in the root directory with the following variables:

- `OPENAI_API_KEY`: Your OpenAI API key (required)

## Security Note

Never commit your `.env` file or expose your API keys. The `.env` file is included in `.gitignore` to prevent accidental commits.

## License

[Your chosen license]