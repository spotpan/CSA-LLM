import torch
from torch_geometric.data import Data
import numpy as np
from scipy.sparse import coo_matrix
import openai
import ast  # To safely evaluate literal structures
import time
import difflib
import os
import re

# Load the .pt file
data = torch.load("/home/user/Documents/LLMGNN-master/data/citeseer_fixed_sbert.pt")

# Extract edge_index and select edges with odd indices
edge_index = data['edge_index']

edge_index = edge_index[0]

source_node = edge_index[1::2] #Even indices
target_node = edge_index[::2] # Odd indices
#odd_index_edges = edge_index[:, ::2]

# Number of nodes (assuming nodes are indexed from 0 to num_nodes-1)
num_nodes = data['x'].size(0)

# Create the adjacency matrix
adjacency_matrix = coo_matrix((np.ones(len(source_node)), (source_node, target_node)),
                              shape=(num_nodes, num_nodes))

# Convert the adjacency matrix to CSR format for efficient row slicing
adjacency_matrix_csr = adjacency_matrix.tocsr()

# Function to get predecessors and successors
def get_neighbors(adjacency_matrix, node):
    # Successors
    successors = adjacency_matrix.indices[adjacency_matrix.indptr[node]:adjacency_matrix.indptr[node+1]]

    # Predecessors
    predecessors = []
    for idx in range(adjacency_matrix.shape[0]):  # Iterate over all nodes
        if node in adjacency_matrix.indices[adjacency_matrix.indptr[idx]:adjacency_matrix.indptr[idx+1]]:
            predecessors.append(idx)

    return predecessors, successors

# Function to get the first 30 words of a text
"""
def get_first_30_words(text):
    return ' '.join(text.split()[:30])

"""
# Function to get the first 30 words of a text
def get_first_30_words(text):
    words = text.split()
    first_30_words = ' '.join(words[:30])
    return re.sub(r'[^\w\s]', '', first_30_words)  # Clean text to remove special characters


# Categories for the guessing task
categories = [
    "Agents", "Machine Learning", "Information Retrieval",
    "Database", "Human Computer Interaction", "Artificial Intelligence"
]

# Mapping from category name to label index
category_to_label = {
    'Agents': 0, 'Machine Learning': 1, 'Information Retrieval': 2,
    'Database': 3, 'Human Computer Interaction': 4, 'Artificial Intelligence': 5
}

# Initialize OpenAI API key (replace 'your-api-key' with your actual API key)
openai.api_key = "your-api-key"

# Function to query OpenAI API and get predictions
# Function to query OpenAI API and get predictions
def get_openai_predictions(summary, retries=5, delay=2):
    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": summary}
                ]
            )
            answer_text = response.choices[0].message['content'].strip()
            predictions = ast.literal_eval(answer_text)  # Safely evaluate literal structures
            if not isinstance(predictions, list):
                raise ValueError("Predictions is not a list")
            for pred in predictions:
                if not isinstance(pred, dict) or 'answer' not in pred or 'confidence' not in pred:
                    raise ValueError("Predictions format is invalid")
            return predictions
        except ValueError as ve:
            print(f"Error parsing predictions on attempt {attempt + 1}: {ve}")
            time.sleep(delay)
        except openai.error.APIError as api_err:
            print(f"OpenAI API error on attempt {attempt + 1}: {api_err}")
            if api_err.status_code == 429:  # Handle rate limit (too many requests)
                delay *= 2  # Exponential backoff
            time.sleep(delay)
        except Exception as e:
            print(f"Unexpected error on attempt {attempt + 1}: {e}")
            time.sleep(delay)
    return None

# Function to normalize confidences
def normalize_confidences(predictions):
    for pred in predictions:
        pred['confidence'] = float(pred['confidence'])
    total_confidence = sum(pred['confidence'] for pred in predictions)
    if total_confidence != 100:
        for pred in predictions:
            pred['confidence'] = pred['confidence'] * 100 / total_confidence
    return predictions

# Function to find the best match for a given category
def find_best_match(category, categories):
    match = difflib.get_close_matches(category, categories, n=1, cutoff=0.6)
    return match[0] if match else None

# Initialize lists to store predictions and confidences
pred_labels = []
confidences = []

# Default category and confidence if no valid categories are found
default_category = "Artificial Intelligence"
default_confidence = 0.0

# Directory to save results
results_dir = 'citeseer_results'
os.makedirs(results_dir, exist_ok=True)

# Iterate over each node and query OpenAI API for predictions
for node in range(num_nodes):
    raw_text = get_first_30_words(data['raw_texts'][node])
    predecessors, successors = get_neighbors(adjacency_matrix_csr, node)

    summary = f"The content of the paper is '{raw_text}'"
    
    
    if len(predecessors) > 0:
        pred_texts = [get_first_30_words(data['raw_texts'][pred]) for pred in predecessors]
        summary += f", which is cited by the paper(s) that {', '.join(pred_texts)}"

    if len(successors) > 0:
        succ_texts = [get_first_30_words(data['raw_texts'][succ]) for succ in successors]
        summary += f", and cites the paper(s) that {', '.join(succ_texts)}"
    


    summary += ".\n\n"

    # Add category guessing task placeholder
    summary += "What's the category of this paper? Provide your 6 best guesses and a confidence number that each is correct (0 to 100) for the following question from most probable to least. The sum of all confidence should be 100. "
    summary += "[{'answer': 'Agents', 'confidence': <confidence score>}, {'answer': 'Machine Learning', 'confidence': <confidence score>}, {'answer': 'Information Retrieval', 'confidence': <confidence score>}, {'answer': 'Database', 'confidence': <confidence score>}, {'answer': 'Human Computer Interaction', 'confidence': <confidence score>}, {'answer': 'Artificial Intelligence', 'confidence': <confidence score>}]"
    summary += "Please answer with this format above directly"
 

    # Query OpenAI API
    predictions = get_openai_predictions(summary)

    #print(summary)

    #print("\n")

    #print(predictions)

    # Ensure valid predictions
    if predictions is None:
        print(f"No valid predictions for node {node}")
        pred_labels.append(category_to_label[default_category])
        confidences.append(default_confidence)
        continue

    # Normalize the confidences
    predictions = normalize_confidences(predictions)

    # Find the best match for each prediction
    valid_predictions = []
    for pred in predictions:
        category = pred.get('answer', pred.get('category', '')).replace(" ", "_")
        best_match = find_best_match(category, categories)
        if best_match:
            pred['answer'] = best_match
            valid_predictions.append(pred)
        else:
            print(f"Node {node}: No close match found for category '{category}'")

    if not valid_predictions:
        print(f"No valid predictions for node {node}")
        pred_labels.append(category_to_label[default_category])
        confidences.append(default_confidence)
        continue

    # Find the predicted label with the highest confidence
    max_confidence = -1
    max_label = -1
    for pred in valid_predictions:
        category = pred['answer']
        label_index = category_to_label[category]
        #print(f"Node {node}: Checking category '{category}' with label index {label_index} and confidence {pred['confidence']}")
        if pred['confidence'] > max_confidence:
            max_confidence = pred['confidence']
            max_label = label_index

    print("max_label:",max_label,"max_confidence",max_confidence)
    pred_labels.append(max_label)
    confidences.append(max_confidence)

    # Save results for every 100 nodes processed
    if (node + 1) % 100 == 0:
        pred_labels_tensor = torch.tensor(pred_labels)
        confidences_tensor = torch.tensor(confidences)
        torch.save({'pred': pred_labels_tensor, 'conf': confidences_tensor}, os.path.join(results_dir, f'results_{node + 1}.pt'))

# Save final results
pred_labels_tensor = torch.tensor(pred_labels)
confidences_tensor = torch.tensor(confidences)
torch.save({'pred': pred_labels_tensor, 'conf': confidences_tensor}, os.path.join(results_dir, 'final_results.pt'))

print("Prediction and confidence results saved successfully.")
