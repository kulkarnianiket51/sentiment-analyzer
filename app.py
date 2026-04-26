from transformers import pipeline
import gradio as gr

# Load pretrained model explicitly
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Define function
def analyze_sentiment(text):
    result = classifier(text)[0]
    return f"{result['label']} (confidence: {result['score']:.2f})"

# Create Gradio UI
demo = gr.Interface(
    fn=analyze_sentiment,
    inputs="text",
    outputs="text",
    title="Sentiment Analyzer",
    description="Enter text and see if it's positive or negative."
)

# Launch app
if __name__ == "__main__":
    demo.launch()
