from pathlib import Path
import json
import pickle
import string

import numpy as np
import torch

from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification
)

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from captum.attr import LayerIntegratedGradients


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

BIOBERT_DIR = ARTIFACTS_DIR / "biobert"
LABEL_MAPS_PATH = ARTIFACTS_DIR / "label_maps.pkl"
RETRIEVAL_DATA_PATH = ARTIFACTS_DIR / "retrieval_data.pkl"
DISEASE_INFO_PATH = ARTIFACTS_DIR / "disease_info.json"


# ============================================================
# Device
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {DEVICE}")


# ============================================================
# Load label mappings
# ============================================================

with open(LABEL_MAPS_PATH, "rb") as f:
    label_maps = pickle.load(f)

label2id = label_maps["label2id"]
id2label = label_maps["id2label"]


# ============================================================
# Load disease knowledge base
# ============================================================

with open(DISEASE_INFO_PATH, "r", encoding="utf-8") as f:
    disease_info = json.load(f)


# ============================================================
# Load BioBERT tokenizer
# ============================================================

biobert_tokenizer = BertTokenizerFast.from_pretrained(
    str(BIOBERT_DIR),
    do_lower_case=False
)


# ============================================================
# Load BioBERT model
# ============================================================

biobert_model = BertForSequenceClassification.from_pretrained(
    str(BIOBERT_DIR)
)

# Eager attention is required for the Captum attribution setup
biobert_model.set_attn_implementation("eager")

biobert_model.to(DEVICE)
biobert_model.eval()


# ============================================================
# Load Sentence-BERT
# ============================================================

retriever_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ============================================================
# Load retrieval corpus
# ============================================================

with open(RETRIEVAL_DATA_PATH, "rb") as f:
    retrieval_data = pickle.load(f)

retrieval_texts = retrieval_data["texts"]
retrieval_labels = retrieval_data["labels"]
retrieval_embeddings = retrieval_data["embeddings"]


# ============================================================
# Sentence-BERT retrieval
# ============================================================

def retrieve_similar_cases(
    query_text: str,
    top_k: int = 5
):
    query_embedding = retriever_model.encode(
        [query_text],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    similarities = cosine_similarity(
        query_embedding,
        retrieval_embeddings
    )[0]

    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []

    for idx in top_indices:
        results.append({
            "symptom_text": retrieval_texts[idx],
            "disease_label": retrieval_labels[idx],
            "similarity": float(similarities[idx])
        })

    return results


# ============================================================
# Integrated Gradients
# ============================================================

def model_forward(
    input_ids,
    attention_mask,
    token_type_ids
):
    outputs = biobert_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids
    )

    return outputs.logits


lig = LayerIntegratedGradients(
    model_forward,
    biobert_model.bert.embeddings.word_embeddings
)


# ============================================================
# XAI display filtering
# ============================================================

XAI_STOPWORDS = {
    "i", "im", "ive", "my", "me", "mine",
    "we", "were", "weve", "our", "ours",
    "you", "your", "yours",
    "he", "she", "they", "their",

    "the", "a", "an",
    "and", "or", "but",
    "to", "of", "from", "for", "with",
    "in", "on", "at", "by",

    "this", "that", "these", "those",
    "it", "its",

    "is", "am", "are", "was", "were",
    "be", "been", "being",

    "have", "has", "had",
    "do", "does", "did",

    "can", "could", "would", "should",

    "very", "really", "quite",
    "some", "any",

    "when", "whenever",
    "while", "also",

    "ve", "re", "ll", "d", "m", "s", "t",

    "lot", "lately"
}


# ============================================================
# Integrated Gradients explanation
# ============================================================

def explain_symptoms(
    text: str,
    n_steps: int = 256
):
    biobert_model.eval()

    # --------------------------------------------------------
    # 1. Tokenization
    # --------------------------------------------------------

    encoded = biobert_tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128
    )

    input_ids = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded["attention_mask"].to(DEVICE)
    token_type_ids = encoded["token_type_ids"].to(DEVICE)

    # --------------------------------------------------------
    # 2. Prediction
    # --------------------------------------------------------

    with torch.no_grad():
        logits = biobert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        ).logits

    probabilities = torch.softmax(
        logits,
        dim=-1
    )

    predicted_id = torch.argmax(
        probabilities,
        dim=-1
    ).item()

    # --------------------------------------------------------
    # 3. PAD baseline
    # Keep CLS and SEP unchanged
    # --------------------------------------------------------

    baseline_ids = input_ids.clone()

    cls_id = biobert_tokenizer.cls_token_id
    sep_id = biobert_tokenizer.sep_token_id
    pad_id = biobert_tokenizer.pad_token_id

    special_mask = (
        (baseline_ids == cls_id) |
        (baseline_ids == sep_id)
    )

    baseline_ids[~special_mask] = pad_id

    # --------------------------------------------------------
    # 4. Layer Integrated Gradients
    # --------------------------------------------------------

    attributions, delta = lig.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        additional_forward_args=(
            attention_mask,
            token_type_ids
        ),
        target=predicted_id,
        n_steps=n_steps,
        method="gausslegendre",
        return_convergence_delta=True,
        internal_batch_size=8
    )

    token_scores = (
        attributions
        .sum(dim=-1)
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # 5. Completeness / convergence check
    # --------------------------------------------------------

    with torch.no_grad():

        original_output = biobert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        ).logits[0, predicted_id].item()

        baseline_output = biobert_model(
            input_ids=baseline_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        ).logits[0, predicted_id].item()

    output_difference = (
        original_output -
        baseline_output
    )

    relative_delta = (
        abs(delta.item()) /
        max(abs(output_difference), 1e-8)
    )

    # --------------------------------------------------------
    # 6. Convert WordPieces → readable words
    # --------------------------------------------------------

    tokens = biobert_tokenizer.convert_ids_to_tokens(
        input_ids.squeeze(0)
    )

    merged_words = []
    merged_scores = []

    current_word = ""
    current_score = 0.0

    for token, score in zip(
        tokens,
        token_scores
    ):

        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue

        if token.startswith("##"):
            current_word += token[2:]
            current_score += score

        else:

            if current_word:
                merged_words.append(current_word)
                merged_scores.append(current_score)

            current_word = token
            current_score = score

    if current_word:
        merged_words.append(current_word)
        merged_scores.append(current_score)

    # --------------------------------------------------------
    # 7. Clean explanation terms
    # --------------------------------------------------------

    clean_attributions = []

    for word, score in zip(
        merged_words,
        merged_scores
    ):

        word = word.strip()
        normalized = word.lower()

        # Skip punctuation-only tokens
        if word in string.punctuation:
            continue

        # Skip generic words / fragments
        if normalized in XAI_STOPWORDS:
            continue

        # Skip tokens without alphabetic characters
        if not any(char.isalpha() for char in word):
            continue

        clean_attributions.append(
            (word, float(score))
        )

    # --------------------------------------------------------
    # 8. Positive / negative contributors
    # --------------------------------------------------------

    top_positive = sorted(
        [
            (word, score)
            for word, score in clean_attributions
            if score > 0
        ],
        key=lambda x: x[1],
        reverse=True
    )[:10]

    top_negative = sorted(
        [
            (word, score)
            for word, score in clean_attributions
            if score < 0
        ],
        key=lambda x: x[1]
    )[:10]

    return {
        "top_positive": top_positive,
        "top_negative": top_negative,
        "convergence_delta": float(delta.item()),
        "relative_delta": float(relative_delta),
        "relative_delta_percent": float(
            relative_delta * 100
        ),
        "integration_steps": n_steps
    }


# ============================================================
# Disease prediction
# ============================================================

def predict_disease(
    text: str
):
    encoded = biobert_tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128
    )

    input_ids = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded["attention_mask"].to(DEVICE)
    token_type_ids = encoded["token_type_ids"].to(DEVICE)

    with torch.no_grad():

        logits = biobert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        ).logits

    probabilities = torch.softmax(
        logits,
        dim=-1
    )

    top_n = min(
        3,
        probabilities.shape[-1]
    )

    top_probs, top_ids = torch.topk(
        probabilities,
        k=top_n,
        dim=-1
    )

    top_predictions = []

    for prob, class_id in zip(
        top_probs[0].cpu().tolist(),
        top_ids[0].cpu().tolist()
    ):

        top_predictions.append({
            "disease": id2label[class_id],
            "probability": float(prob),
            "probability_percent": float(
                prob * 100
            )
        })

    predicted = top_predictions[0]

    return {
        "prediction": predicted["disease"],
        "confidence": predicted["probability"],
        "confidence_percent": predicted[
            "probability_percent"
        ],
        "top_predictions": top_predictions
    }


# ============================================================
# Complete inference pipeline
# ============================================================

def predict_case(
    text: str,
    top_k: int = 5,
    xai_steps: int = 256
):
    """
    Complete project inference pipeline:

    1. BioBERT classification
    2. Top-3 predictions
    3. Integrated Gradients explanation
    4. Sentence-BERT retrieval
    5. Disease knowledge base
    """

    # --------------------------------------------------------
    # BioBERT prediction
    # --------------------------------------------------------

    prediction_result = predict_disease(text)

    predicted_label = prediction_result["prediction"]

    # --------------------------------------------------------
    # XAI
    # --------------------------------------------------------

    explanation = explain_symptoms(
        text,
        n_steps=xai_steps
    )

    # --------------------------------------------------------
    # Similar historical cases
    # --------------------------------------------------------

    similar_cases = retrieve_similar_cases(
        text,
        top_k=top_k
    )

    # --------------------------------------------------------
    # Knowledge base
    # --------------------------------------------------------

    knowledge = disease_info.get(
        predicted_label,
        {
            "description": "No additional information available.",
            "common_symptoms": [],
            "source": None
        }
    )

    # --------------------------------------------------------
    # Stable result structure
    # --------------------------------------------------------

    return {
        "input_text": text,

        "prediction": predicted_label,

        "confidence": prediction_result[
            "confidence"
        ],

        "confidence_percent": prediction_result[
            "confidence_percent"
        ],

        "top_predictions": prediction_result[
            "top_predictions"
        ],

        "explanation": explanation,

        "similar_cases": similar_cases,

        "disease_information": knowledge
    }


# ============================================================
# Local test
# ============================================================

if __name__ == "__main__":

    sample_text = (
        "I've been suffering from severe constipation lately, "
        "and whenever I do go to the restroom, it hurts a lot. "
        "Aside from that, my anus has been really itchy, "
        "and I've observed some blood in my stool."
    )

    print("\n" + "=" * 60)
    print("SYMPTOM-TO-DIAGNOSIS CDSS")
    print("=" * 60)

    result = predict_case(
        sample_text,
        top_k=3,
        xai_steps=256
    )

    print("\nPrediction:")
    print(result["prediction"])

    print("\nConfidence:")
    print(
        f"{result['confidence_percent']:.2f}%"
    )

    print("\nTop 3 predictions:")

    for item in result["top_predictions"]:
        print(
            f"{item['disease']:35s}"
            f"{item['probability_percent']:.2f}%"
        )

    print("\nTop positive contributors:")

    for word, score in result[
        "explanation"
    ]["top_positive"][:5]:

        print(
            f"{word:20s}"
            f"{score:+.6f}"
        )

    print("\nTop negative contributors:")

    for word, score in result[
        "explanation"
    ]["top_negative"][:5]:

        print(
            f"{word:20s}"
            f"{score:+.6f}"
        )

    print("\nXAI relative delta:")

    print(
        f"{result['explanation']['relative_delta_percent']:.2f}%"
    )

    print("\nSimilar cases:")

    for case in result[
        "similar_cases"
    ]:

        print(
            f"{case['disease_label']:35s}"
            f"{case['similarity']:.4f}"
        )

    print("\nKnowledge base:")

    print(
        result["disease_information"]
    )