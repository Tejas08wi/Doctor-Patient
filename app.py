import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objs as go
from collections import Counter

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Define concern and reassurance keywords
concern_keywords = {'error', 'wrong dosage', 'issue', 'terrible', 'symptoms', 'worst', 'test result', 'abnormality', 
                   'shortness of breath', 'panic', 'dangerous', 'breathing difficulty', 'emergency', 'anxious', 
                   'blurred vision', 'unconscious', 'crying', 'abnormal levels', 'problem', 'irregular', 'chest pain', 
                   'worse', 'failure', 'abnormal', 'critical', 'immediate', 'bleeding', 'severe', 'unbearable', 
                   'weakness', 'deteriorating', 'overdose', 'fainting', 'fear', 'swelling', 'fatigue', 'worried', 
                   'ineffective', 'life-threatening', 'side effect', 'rash', 'urgent'}

reassurance_keywords = {"better", "normal", "okay", "fine", "don't worry", "understand",
                       "help", "support", "improving", "safe"}

# Load the dataset with error handling
try:
    new_df = pd.read_csv(
        "C:/Users/KIIT/Dash/gpt-4.csv",
        encoding='utf-8',
        nrows=3000,
        on_bad_lines='skip'
    )
except Exception as e:
    print(f"Error loading CSV file: {e}")
    exit()

# Check if required columns exist
if 'conversation' not in new_df.columns:
    print("Error: 'conversation' column not found in the dataset.")
    exit()

# Preprocessing function
lem = WordNetLemmatizer()
stop = set(stopwords.words('english'))
additional_stopwords = {"doctor", "patient", "okay", "thank", "hello", "morning"}
stop.update(additional_stopwords)

def preprocess_text(text):
    tokens = word_tokenize(str(text).lower())
    tokens = [lem.lemmatize(word) for word in tokens if word.isalnum() and word not in stop]
    return ' '.join(tokens)

# Function to calculate risk score
def calculate_risk_score(text):
    text = text.lower()
    concern_count = sum(1 for keyword in concern_keywords if keyword.lower() in text)
    reassurance_count = sum(1 for keyword in reassurance_keywords if keyword.lower() in text)
    return concern_count - reassurance_count

# Apply preprocessing and risk calculation
new_df['cleaned_text'] = new_df['conversation'].apply(preprocess_text)
new_df['risk_score'] = new_df['conversation'].apply(calculate_risk_score)
new_df['risk_level'] = pd.qcut(new_df['risk_score'], q=3, labels=['Low', 'Medium', 'High'])

# Extract common symptoms and diseases
common_symptoms = ['pain', 'fever', 'cough', 'headache', 'nausea', 'fatigue']
common_diseases = ['diabetes', 'hypertension', 'asthma', 'arthritis', 'depression']

def count_terms(text, term_list):
    return sum(1 for term in term_list if term in text.lower())

new_df['symptom_count'] = new_df['conversation'].apply(lambda x: count_terms(x, common_symptoms))
new_df['disease_count'] = new_df['conversation'].apply(lambda x: count_terms(x, common_diseases))

# Generate word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(new_df['cleaned_text']))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.savefig('wordcloud.png', bbox_inches='tight', dpi=300)

# Dashboard layout
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Medical Conversations Analysis Dashboard", 
            style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '30px'}),
    
    # Key Insights Summary
    html.Div([
        html.H2("Key Insights", style={'textAlign': 'center', 'color': '#34495e'}),
        html.Div([
            html.Div([
                html.H3("Total Conversations"),
                html.H4(len(new_df)),
            ], className='summary-card'),
            html.Div([
                html.H3("High Risk Patients"),
                html.H4(len(new_df[new_df['risk_level'] == 'High'])),
            ], className='summary-card'),
            html.Div([
                html.H3("Average Risk Score"),
                html.H4(f"{new_df['risk_score'].mean():.1f}"),
            ], className='summary-card'),
            html.Div([
                html.H3("Critical Cases"),
                html.H4(len(new_df[new_df['risk_score'] > new_df['risk_score'].quantile(0.9)])),
            ], className='summary-card'),
        ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '30px'}),
    ]),

    # Analysis Controls
    # Analysis Controls
html.Div([
    html.Div([
        html.Label("Select Analysis Type:"),
        dcc.Dropdown(
            id='analysis-type',
            options=[
                {'label': 'Word Cloud', 'value': 'wordcloud'},
                {'label': 'Risk Distribution', 'value': 'risk'},
                {'label': 'Symptom Analysis', 'value': 'symptoms'},
                {'label': 'Disease Trends', 'value': 'diseases'},
                {'label': 'Conversation Patterns', 'value': 'patterns'}
            ],
            value='risk',
            style={'width': '100%'}
        ),
    ], style={'width': '30%', 'display': 'inline-block'}),
    
    html.Div([
        html.Label("Dataset Size:"),
        dcc.RangeSlider(
            id='time-slider',
            min=0,
            max=len(new_df),
            value=[0, len(new_df)],
            marks={0: 'Start', len(new_df): 'End'}
        ),
    ], style={'width': '60%', 'display': 'inline-block', 'marginLeft': '20px'})
], style={'marginBottom': '20px'}),

    # Main Content Area
    html.Div([
        html.Div([
            html.Div(id='main-visualization'),
        ], style={'width': '70%', 'display': 'inline-block'}),

        html.Div([
            html.H3("Quick Statistics"),
            html.Div(id='detail-stats'),
        ], style={'width': '28%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '2%'})
    ]),
    
    html.Div([
        html.H3("Key Findings Summary"),
        html.Div(id='findings-summary'),
    ], style={'marginTop': '30px'})
])

@app.callback(
    [Output('main-visualization', 'children'),
     Output('detail-stats', 'children'),
     Output('findings-summary', 'children')],
    [Input('analysis-type', 'value'),
     Input('time-slider', 'value')]
)
def update_visualization(analysis_type, time_range):
    filtered_df = new_df.iloc[time_range[0]:time_range[1]]
    
    main_viz = []
    stats = []
    findings = []
    
    if analysis_type == 'wordcloud':
        main_viz = html.Img(src='wordcloud.png', style={'width': '100%'})
        stats = [
            html.P(f"Most common word: {pd.Series(' '.join(filtered_df['cleaned_text']).split()).value_counts().index[0]}"),
            html.P(f"Unique words: {len(set(' '.join(filtered_df['cleaned_text']).split()))}")
        ]
    
    elif analysis_type == 'risk':
        fig = px.histogram(filtered_df, x='risk_score', color='risk_level',
                          title='Patient Risk Score Distribution',
                          labels={'risk_score': 'Risk Score', 'count': 'Number of Patients'})
        main_viz = dcc.Graph(figure=fig)
        stats = [
            html.P(f"High Risk Patients: {len(filtered_df[filtered_df['risk_level'] == 'High'])}"),
            html.P(f"Average Risk Score: {filtered_df['risk_score'].mean():.1f}")
        ]
        
    elif analysis_type == 'symptoms':
        symptom_counts = {symptom: sum(filtered_df['conversation'].str.contains(symptom, case=False)) 
                         for symptom in common_symptoms}
        fig = px.bar(x=list(symptom_counts.keys()), y=list(symptom_counts.values()),
                    title='Symptom Frequency Analysis')
        main_viz = dcc.Graph(figure=fig)
        stats = [html.P(f"Most common symptom: {max(symptom_counts, key=symptom_counts.get)}")]
        
    elif analysis_type == 'diseases':
        disease_counts = {disease: sum(filtered_df['conversation'].str.contains(disease, case=False)) 
                         for disease in common_diseases}
        fig = px.pie(values=list(disease_counts.values()), names=list(disease_counts.keys()),
                    title='Disease Distribution')
        main_viz = dcc.Graph(figure=fig)
        stats = [html.P(f"Most mentioned disease: {max(disease_counts, key=disease_counts.get)}")]
        
    elif analysis_type == 'patterns':
        conversation_lengths = filtered_df['conversation'].str.len()
        fig = px.histogram(conversation_lengths, title='Conversation Length Distribution')
        main_viz = dcc.Graph(figure=fig)
        stats = [
            html.P(f"Average conversation length: {int(conversation_lengths.mean())} characters"),
            html.P(f"Longest conversation: {int(conversation_lengths.max())} characters")
        ]
    
    findings = html.Div([
        html.P("Key Findings:"),
        html.Ul([
            html.Li(f"Analyzed {len(filtered_df)} conversations"),
            html.Li(f"High Risk Patients: {len(filtered_df[filtered_df['risk_level'] == 'High'])}"),
            html.Li(f"Critical Cases: {len(filtered_df[filtered_df['risk_score'] > filtered_df['risk_score'].quantile(0.9)])}")
        ])
    ])
    
    return main_viz, stats, findings

if __name__ == '__main__':
    app.run_server(debug=True)
