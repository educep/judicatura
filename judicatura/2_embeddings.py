"""
Created by Analitika at 09/08/2024
contact@analitika.fr
"""

# External imports
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer
from judicatura.helpers import (
    load_sentences,
    compute_embeddings,
    query_embeddings,
    do_radar_plot,
    do_scatter_plot,
)

# Internal imports


# Load tokenizer and model
model_str = "distilbert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_str)
model = AutoModel.from_pretrained(model_str)


text_reference = """
El derecho a la tutela judicial efectiva garantiza que toda persona pueda acceder a los tribunales para la protección de sus derechos fundamentales sin restricciones arbitrarias.
"""
text_reference_embeddings = compute_embeddings(text_reference, tokenizer, model)

# Load sentences
sentences_df = load_sentences(file_name="judicial_sentences.json")

# Compute embeddings and similarities for each sentence in the DataFrame
all_embeddings = []
all_labels = []
results = pd.DataFrame(columns=sentences_df.columns)
for column in sentences_df.columns:
    for idx, sentence in enumerate(sentences_df[column]):
        sentence_embeddings = compute_embeddings(sentence, tokenizer, model)
        # Store embeddings
        all_embeddings.append(sentence_embeddings)
        all_labels.append(column)
        similarity = cosine_similarity([text_reference_embeddings], [sentence_embeddings])[0][0]
        results.loc[idx, column] = similarity

# annotation = text_reference.replace("información, comprender", "información,<br>comprender")
do_radar_plot(results, sentences_df, "")

# Convert embeddings list to a NumPy array
all_embeddings = np.array(all_embeddings)


from umap import UMAP
from sklearn.preprocessing import MinMaxScaler
# Scale features to [0,1] range
X_scaled = MinMaxScaler().fit_transform(all_embeddings)
# Initialize and fit UMAP
mapper = UMAP(n_components=2, metric="cosine", random_state=42).fit(X_scaled)
# Create a DataFrame of 2D embeddings
df_emb = pd.DataFrame(mapper.embedding_, columns=["X", "Y"])
df_emb["label"] = all_labels

do_scatter_plot(df_emb)




# todo: apply also with queries

# Define the source sentence and a list of other sentences
source_sentence = "The answer to the universe is 42."
sentences = [
    "Life, the universe, and everything.",
    "What is the meaning of life?"
]

# Prepare the payload correctly
payload = {
    "inputs": {
        "source_sentence": source_sentence,
        "sentences": sentences
    }
}


data = query_embeddings({"inputs": "El universo es un gran [MASK]."})
print(data)












