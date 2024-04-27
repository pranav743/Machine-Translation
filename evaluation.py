import pandas as pd
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize

# Load the Excel file
df = pd.read_excel('translation_data-1.xlsx')

def clean_text(text):
    text = text.strip()
    return text.lower()

# Function to calculate BLEU score
def calculate_bleu_score(actual, predicted):
    actual_tokens = word_tokenize(actual)
    predicted_tokens = word_tokenize(predicted)
    return sentence_bleu([actual_tokens], predicted_tokens, (0.5,0.5))

# Calculate BLEU scores for each row and update the BLEU Score column
bleu_scores_lstm = []
bleu_scores_t5 = []
bleu_scores_transformer = []

for index, row in df.iterrows():
    actual = clean_text(str(row['Actual']))
    lstm_predicted = clean_text(str(row['LSTM-Prediction']))
    t5_predicted = clean_text(str(row['T5-Prediction']))
    transformer_predicted = clean_text(str(row['Transformer-Prediction']))
    bleu_score_lstm = calculate_bleu_score(actual, lstm_predicted)
    bleu_score_t5 = calculate_bleu_score(actual, t5_predicted)
    bleu_score_transformer = calculate_bleu_score(actual, transformer_predicted)
    bleu_scores_lstm.append(bleu_score_lstm)
    bleu_scores_t5.append(bleu_score_t5)
    bleu_scores_transformer.append(bleu_score_transformer)

df['BLEU Score - LSTM'] = bleu_scores_lstm
df['BLEU Score - T5'] = bleu_scores_t5
df['BLEU Score - Transformer'] = bleu_scores_transformer

# Save the updated DataFrame to the Excel file
df.to_excel('output_comparison_ta-hi.xlsx', index=False)
