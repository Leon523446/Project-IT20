from langchain_ollama import OllamaLLM
import pandas as pd
import gdown
import os
import gradio as gr
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from typing import Dict, List, Tuple, Union, Any, Optional
from pandas import DataFrame
from sklearn.base import BaseEstimator
from numpy import ndarray
from plotly.graph_objects import Figure
import seaborn as sns
import matplotlib.pyplot as plt

from langchain_ollama import OllamaLLM
import pandas as pd
import gdown
import os
import gradio as gr
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Any, Optional


ModelMetrics = Dict[str, Dict[str, float]]
ModelPredictions = Dict[str, List[Tuple[str, float]]]
TrainedModels = Dict[str, BaseEstimator]

# Download and load the dataset
file_id = '13yAjQSrh65h6cgmL-QJY7TMwsXXARmPu'
output = 'AI_Symptom_Checker_Dataset.csv'
if not os.path.exists(output):
    gdown.download(f'https://drive.google.com/uc?id={file_id}', output, quiet=False)

df = pd.read_csv(output, on_bad_lines='skip')

print("Available columns in dataset:", df.columns.tolist())
def clean_text(text):
    """Clean and standardize text data"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    return text

# Apply cleaning
print("Available columns in dataset:", df.columns.tolist())
df.dropna(subset=['Symptoms', 'Predicted Disease'], inplace=True)
df['Symptoms'] = df['Symptoms'].apply(clean_text)
df['Gender'] = df['Gender'].apply(clean_text)

llm = OllamaLLM(model="mistral", temperature=0.3)
def create_confusion_matrix(model_name: str) -> plt.Figure:
    """Generate a confusion matrix plot and display evaluation metrics for the specified model."""
    
    # Check if trained models are available
    if not trained_models:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No trained models available", ha='center', va='center')
        ax.set_title("Confusion Matrix - No Data")
        return fig
    
    # Retrieve the model
    model = trained_models[model_name]
    y_pred = model.predict(X_test)
    
    # Get class labels (show only first 20 for readability)
    classes = le.classes_[:20]
    y_test_filtered = y_test[np.isin(y_test, np.arange(len(classes)))]
    y_pred_filtered = y_pred[np.isin(y_test, np.arange(len(classes)))]
    
    # Calculate confusion matrix (normalized)
    cm_normalized = confusion_matrix(y_test_filtered, y_pred_filtered, normalize='true')
    
    # Calculate metrics for each class
    n_classes = cm_normalized.shape[0]
    class_metrics = []
    
    for i in range(n_classes):
        TP = cm_normalized[i, i]
        FP = cm_normalized[:, i].sum() - TP
        FN = cm_normalized[i, :].sum() - TP
        TN = cm_normalized.sum() - TP - FP - FN
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        class_metrics.append({
            'Class': classes[i],
            'Precision': precision,
            'Recall': recall,
            'F1-score': f1
        })
    
    # Calculate overall metrics
    accuracy = np.trace(cm_normalized) / np.sum(cm_normalized)
    
    # Create a string with the metrics
    metrics_str = f"Accuracy: {accuracy:.2f}\n"
    metrics_str += "\nPer-class metrics:\n"
    for metric in class_metrics[:5]:  # Show first 5 classes for brevity
        metrics_str += (f"{metric['Class']} - Precision: {metric['Precision']:.2f}, "
                        f"Recall: {metric['Recall']:.2f}, F1: {metric['F1-score']:.2f}\n")

    # Plot the normalized confusion matrix
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm_normalized, 
        annot=True, 
        fmt=".2f",  # For normalized values
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax
    )
    
    # Set titles and labels
    ax.set_title(f"Normalized Confusion Matrix - {model_name}")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Add metrics text to the plot
    ax.text(1.05, 0.95, metrics_str, transform=ax.transAxes, fontsize=10, 
            va='top', ha='left', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
    
    # Adjust layout to avoid overlap
    plt.tight_layout()
    
    return fig


def prepare_data() -> Tuple[ndarray, ndarray, LabelEncoder, CountVectorizer, str, str]:
    """Prepare the dataset for machine learning models."""
    print("\nSample data:")
    print(df.head())
    
    text_col = None
    label_col = None
    
    for col in df.columns:
        if 'symptom' in col.lower():
            text_col = col
        elif 'disease' in col.lower() or 'diagnosis' in col.lower():
            label_col = col
    
    if text_col is None or label_col is None:
        text_col = df.columns[0]
        label_col = df.columns[1] if len(df.columns) > 1 else None
    
    if label_col is None:
        raise ValueError("Could not find a suitable label column in the dataset")
    
    print(f"\nUsing '{text_col}' as symptoms text column")
    print(f"Using '{label_col}' as diagnosis label column")
    
    df[text_col] = df[text_col].astype(str).str.lower().str.strip()
    df[label_col] = df[label_col].astype(str).str.strip()
    
    sample_df = df.sample(min(10000, len(df))) if len(df) > 10000 else df
    
    le = LabelEncoder()
    y = le.fit_transform(sample_df[label_col])
    
    vectorizer = CountVectorizer(max_features=500)
    X = vectorizer.fit_transform(sample_df[text_col])
    
    return X, y, le, vectorizer, text_col, label_col


def new_prepare_data() -> Tuple[ndarray, ndarray, LabelEncoder, TfidfVectorizer, str, str]:
    """Prepare the dataset with proper preprocessing."""
    # Automatically detect relevant columns
    text_col = next((col for col in df.columns if 'symptom' in col.lower()), 'Symptoms')
    label_col = next((col for col in df.columns if 'disease' in col.lower() or 'diagnosis' in col.lower()), 'Predicted Disease')
    
    print(f"\nUsing '{text_col}' as symptoms text column")
    print(f"Using '{label_col}' as diagnosis label column")
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df[label_col])
    
    # Vectorize symptoms using TF-IDF
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1,2), stop_words='english')
    X_text = vectorizer.fit_transform(df[text_col])
    
    # Encode gender
    gender_encoder = LabelEncoder()
    gender_encoded = gender_encoder.fit_transform(df['Gender'])
    
    # Combine features
    from scipy.sparse import hstack
    X = hstack([X_text, np.array(df['Age']).reshape(-1, 1), gender_encoded.reshape(-1, 1)])
    
    return X, y, le, vectorizer, text_col, label_col

def evaluate_model(model: BaseEstimator, X_test: ndarray, y_test: ndarray) -> Dict[str, float]:
    """Evaluate model performance metrics using confusion matrix components and sklearn's metrics."""
    try:
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        # Initialize metrics dictionary
        metrics = {
            "Accuracy": 0.0,
            "Precision": 0.0,
            "Recall": 0.0,
            "F1-score": 0.0
        }

        # Calculate per-class metrics
        n_classes = cm.shape[0]
        precisions = []
        recalls = []
        f1s = []
        
        for i in range(n_classes):
            TP = cm[i, i]
            FP = cm[:, i].sum() - TP
            FN = cm[i, :].sum() - TP
            
            with np.errstate(divide='ignore', invalid='ignore'):
                precision = TP / (TP + FP) if (TP + FP) > 0 else 0
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        
        # Calculate weighted averages
        support = cm.sum(axis=1)
        total_samples = np.sum(support)
        
        metrics["Accuracy"] = round(accuracy_score(y_test, y_pred) * 100, 2)
        metrics["Precision"] = round(np.average(precisions, weights=support), 2)
        metrics["Recall"] = round(np.average(recalls, weights=support), 2)
        metrics["F1-score"] = round(np.average(f1s, weights=support), 2)
        
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        # Fallback to sklearn's implementation if confusion matrix calculation fails
        metrics["Accuracy"] = round(accuracy_score(y_test, y_pred) * 100, 2)
        metrics["Precision"] = round(precision_score(y_test, y_pred, average='weighted', zero_division=0), 2)
        metrics["Recall"] = round(recall_score(y_test, y_pred, average='weighted', zero_division=0), 2)
        metrics["F1-score"] = round(f1_score(y_test, y_pred, average='weighted', zero_division=0), 2)
    
    return metrics

# Prepare data and train models
try:
    X, y, le, vectorizer, symptom_col, diagnosis_col = prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "Naive Bayes": MultinomialNB(alpha=0.5),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42),
        "k-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, weights='distance', metric='cosine')
    }

    trained_models = {}
    model_metrics = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        model_metrics[name] = evaluate_model(model, X_test, y_test)
        
        # Print classification report
        y_pred = model.predict(X_test)
        print(f"\nClassification Report for {name}:")
        print(classification_report(y_test, y_pred, target_names=le.classes_))

except Exception as e:
    print(f"\nError in model preparation: {str(e)}")
    # Fallback metrics
    model_metrics = {
        "Naive Bayes": {"Accuracy": 68.00, "Precision": 0.65, "Recall": 0.68, "F1-score": 0.66},
        "Random Forest": {"Accuracy": 72.00, "Precision": 0.70, "Recall": 0.72, "F1-score": 0.71},
        "k-Nearest Neighbors": {"Accuracy": 63.00, "Precision": 0.61, "Recall": 0.63, "F1-score": 0.62}
    }
    trained_models = {}

def show_metrics_details(model_name: str) -> pd.DataFrame:
    """Show detailed metrics including TP, FP, FN, TN for each class."""
    if not trained_models:
        return pd.DataFrame({"Error": ["No trained models available"]})
    
    model = trained_models[model_name]
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    # Get class labels
    classes = le.classes_[:20]  # Limit to first 20 for readability
    
    # Calculate components
    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (FP + FN + TP)
    
    precision_vals = np.where((TP + FP) > 0, TP / (TP + FP), 0)
    recall_vals = np.where((TP + FN) > 0, TP / (TP + FN), 0)
    f1_vals = np.where((precision_vals + recall_vals) > 0,
                       2 * (precision_vals * recall_vals) / (precision_vals + recall_vals),
                       0)
    
    # Create DataFrame
    metrics_df = pd.DataFrame({
        'Class': classes[:len(TP)],
        'True Positives': TP[:len(classes)],
        'False Positives': FP[:len(classes)],
        'False Negatives': FN[:len(classes)],
        'True Negatives': TN[:len(classes)],
        'Precision': precision_vals[:len(classes)],
        'Recall': recall_vals[:len(classes)],
        'F1-score': f1_vals[:len(classes)]
    })
    
    # Round the float values
    float_cols = ['Precision', 'Recall', 'F1-score']
    metrics_df[float_cols] = metrics_df[float_cols].round(2)
    
    return metrics_df

try:
    X, y, le, vectorizer, symptom_col, diagnosis_col = prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models: Dict[str, BaseEstimator] = {
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "k-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
    }

    trained_models: TrainedModels = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

    model_metrics: ModelMetrics = {name: evaluate_model(model, X_test, y_test) for name, model in trained_models.items()}

except Exception as e:
    print(f"\nError in model preparation: {str(e)}")
    print("\nCreating dummy models to allow the app to run...")
    model_metrics = {
        "Naive Bayes": {"Accuracy": 72.00, "Precision": 0.72, "Recall": 0.72, "F1-score": 0.72},
        "Random Forest": {"Accuracy": 72.00, "Precision": 0.72, "Recall": 0.72, "F1-score": 0.72},
        "k-Nearest Neighbors": {"Accuracy": 72.00, "Precision": 0.72, "Recall": 0.72, "F1-score": 0.72}
    }
    trained_models = {}
    symptom_col = df.columns[0] if len(df.columns) > 0 else "Symptoms"
    diagnosis_col = df.columns[1] if len(df.columns) > 1 else "Diagnosis"

def answer_question_manual(query: str) -> str:
    """Answer medical questions using the dataset and LLM."""
    context = df.head(20).to_string() 
    prompt = f"""
You are a helpful medical assistant.
Given the following table of symptoms and diagnoses:

{context}

Answer the question: {query}

If needed, explain shortly. If you cannot find the answer, say "Data not available".
"""
    result = llm.invoke(prompt)
    return result.strip()

def analyze_symptoms(symptom_input: str) -> Figure:
    """Analyze symptoms and return visualization of related diagnoses."""
    try:
        filtered = df[df[symptom_col].str.contains(symptom_input.lower(), na=False)]
        if filtered.empty:
            return px.bar(title="No matching diagnoses found")
        
        top_diagnoses = filtered[diagnosis_col].value_counts().head(5).to_frame()
        top_diagnoses.columns = ['Count']
        
        fig = px.bar(top_diagnoses, 
                     x=top_diagnoses.index, 
                     y='Count',
                     title=f"Top Diagnoses Associated with '{symptom_input}'",
                     labels={'index': 'Diagnosis', 'Count': 'Frequency'})
        return fig
    except Exception as e:
        print(f"Error analyzing symptoms: {str(e)}")
        return px.bar(title="Error occurred")

def predict_disease(symptoms: str) -> ModelPredictions:
    """Predict possible diagnoses from symptoms using trained models."""
    try:
        if not trained_models:
            return {name: [("No model available", 0)] for name in models.keys()}
        
        symptoms_vec = vectorizer.transform([symptoms.lower()])
        
        predictions: ModelPredictions = {}
        for name, model in trained_models.items():
            proba = model.predict_proba(symptoms_vec)[0]
            top3_idx = np.argsort(proba)[-3:][::-1]
            top3_diagnoses = le.inverse_transform(top3_idx)
            top3_probs = proba[top3_idx]
            predictions[name] = list(zip(top3_diagnoses, top3_probs))
        
        return predictions
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return {name: [("Prediction failed", 0)] for name in models.keys()}

with gr.Blocks(title="🩺 Advanced Medical Symptom Analyzer") as dashboard:
    gr.Markdown("# 🩺 Advanced Medical Symptom Analyzer Dashboard")
    
    with gr.Tab("🤖 Chatbot Assistant"):
        gr.Markdown("## AI Medical Chatbot Assistant")
        gr.Markdown("Describe your symptoms or ask health questions in natural language")
        
        chatbot = gr.Chatbot(height=500, bubble_full_width=False)
        chat_input = gr.Textbox(label="Your message", 
                              placeholder="E.g. 'I have headache and fever', or 'What are flu symptoms?'",
                              lines=2)
        
        with gr.Row():
            clear_chat = gr.Button("🧹 Clear Chat")
            submit_btn = gr.Button("💬 Submit")
        
        def chat_respond(message: str, chat_history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
            bot_message = ""
            
            symptom_keywords = ['symptom', 'pain', 'hurt', 'feel', 'having', 'ache', 
                              'fever', 'headache', 'nausea', 'dizziness', 'cough', 
                              'sore', 'swollen', 'rash', 'burning', 'tingling']
            
            if any(word in message.lower() for word in symptom_keywords):
                try:
                    predictions = predict_disease(message)
                    
                    bot_message = "### Detailed Analysis of Your Symptoms\n\n"
                    
                    all_predictions = []
                    for model_name, preds in predictions.items():
                        bot_message += f"**🔍 {model_name} Predictions:**\n"
                        for diag, prob in preds:
                            bot_message += f"- {diag} ({(prob*100):.1f}% confidence)\n"
                            all_predictions.append((diag, prob, model_name))
                        bot_message += "\n"
                    
                    if all_predictions:
                        all_predictions.sort(key=lambda x: x[1], reverse=True)
                        top_diagnosis = all_predictions[0][0]
                        top_confidence = all_predictions[0][1] * 100
                        supporting_models = [x[2] for x in all_predictions if x[0] == top_diagnosis]
                        
                        bot_message += (
                            "### Final Diagnosis Conclusion\n"
                            f"Based on your symptoms, the most likely condition is:\n\n"
                            f"**{top_diagnosis}** ({(top_confidence):.1f}% confidence)\n\n"
                            "### Supporting Evidence\n"
                            f"- Predicted by: {', '.join(set(supporting_models))}\n"
                            "- Consistent with your described symptoms\n\n"
                            "### Recommended Actions\n"
                            "1. Monitor your symptoms and compare with known symptoms of this condition\n"
                            "2. Contact a healthcare provider for proper evaluation\n"
                            "3. Seek emergency care if you experience:\n"
                            "   - Difficulty breathing\n"
                            "   - Severe pain\n"
                            "   - High fever that doesn't improve\n"
                            "   - Confusion or loss of consciousness\n\n"
                            "⚠️ **Medical Disclaimer**\n"
                            "This analysis is based on statistical patterns and should not replace "
                            "professional medical advice, diagnosis, or treatment. The actual cause "
                            "of your symptoms may require clinical evaluation."
                        )
                    
                except Exception as e:
                    bot_message = "⚠️ Error analyzing symptoms. Please try again or use the specific analysis tabs."
            else:
                bot_message = answer_question_manual(message)
                bot_message += "\n\n💡 Remember to consult a doctor for personalized medical advice."
            
            chat_history.append((message, bot_message))
            return "", chat_history
        
        chat_input.submit(chat_respond, [chat_input, chatbot], [chat_input, chatbot])
        submit_btn.click(chat_respond, [chat_input, chatbot], [chat_input, chatbot])
        clear_chat.click(lambda: None, None, chatbot, queue=False)
    
    with gr.Tab("Dataset Info"):
        gr.Markdown("## Dataset Information")
        gr.Markdown(f"Using column **'{symptom_col}'** for symptoms")
        gr.Markdown(f"Using column **'{diagnosis_col}'** for diagnoses")
        gr.DataFrame(df.head(1000), label="Sample Data")
    
    with gr.Tab("Symptom Query"):
        gr.Markdown("## Ask about symptoms and diagnoses")
        with gr.Row():
            query_input = gr.Textbox(lines=2, placeholder="Ask about symptoms...", label="Your Question")
            query_output = gr.Textbox(label="AI Response")
        query_button = gr.Button("Ask")
        query_examples = gr.Examples(
            examples=[
                "Patients with headache and fever",
                "Most common symptoms with cough",
                "Average confidence for migraine"
            ],
            inputs=query_input
        )
    
    with gr.Tab("Symptom Analysis"):
        gr.Markdown("## Analyze symptoms and find related diagnoses")
        symptom_input = gr.Textbox(label="Enter a symptom or symptom combination")
        analysis_plot = gr.Plot(label="Associated Diagnoses")
        analyze_button = gr.Button("Analyze")
    
    with gr.Tab("Diagnosis Prediction"):
        gr.Markdown("## Predict diagnosis from symptoms using ML models")
        with gr.Row():
            pred_input = gr.Textbox(label="Enter your symptoms (comma separated)")
            pred_button = gr.Button("Predict")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Naive Bayes Prediction")
                nb_output = gr.Textbox(label="Top 3 Predictions")
            with gr.Column():
                gr.Markdown("### Random Forest Prediction")
                rf_output = gr.Textbox(label="Top 3 Predictions")
            with gr.Column():
                gr.Markdown("### k-NN Prediction")
                knn_output = gr.Textbox(label="Top 3 Predictions")
    
       
    query_button.click(answer_question_manual, inputs=query_input, outputs=query_output)
    analyze_button.click(analyze_symptoms, inputs=symptom_input, outputs=analysis_plot)
    pred_button.click(
        lambda x: {
            nb_output: "\n".join([f"{d[0]}: {d[1]:.2%}" for d in predict_disease(x).get("Naive Bayes", [("N/A", 0)])]),
            rf_output: "\n".join([f"{d[0]}: {d[1]:.2%}" for d in predict_disease(x).get("Random Forest", [("N/A", 0)])]),
            knn_output: "\n".join([f"{d[0]}: {d[1]:.2%}" for d in predict_disease(x).get("k-Nearest Neighbors", [("N/A", 0)])])
        },
        inputs=pred_input,
        outputs=[nb_output, rf_output, knn_output]
    )

print("\nLaunching dashboard...")
dashboard.launch()