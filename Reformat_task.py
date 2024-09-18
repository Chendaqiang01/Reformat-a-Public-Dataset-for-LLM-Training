import time
from datasets import load_dataset
import json
import re

def extract_qa_pairs(data):
    """
    Extracts question-answer pairs from the input data and saves them in a JSON format.

    Args:
        data: A list of input data items from the dataset.

    Returns:
        A JSON list containing all the extracted question-answer pairs.
    """

    qa_pairs = []  # Initialize an empty list to store question-answer pairs
    id_counter = 0  # Initialize a counter to assign unique IDs to each question-answer pair

    for item in data:
        input_text = item["input"]
        last_answer = "Yes" if item["gold_index"] == 1 else "No"  # Get the answer to the last question
        input_text += last_answer
        # Split each headline using \n\n
        headlines = input_text.split("\n\n")

        for headline in headlines:
            match = re.search(r"(.*?)\?(?!.*?\?) (.*)", headline)

            if match:
                question = match.group(1).strip()
                answer = match.group(2).strip()

                # Create a dictionary to store the question-answer pair
                qa_pair = {
                    "id": id_counter,
                    "Question": question + "?",
                    "Answer": answer
                }

                qa_pairs.append(qa_pair)
                id_counter += 1
    return qa_pairs

# Record the start time
start_time = time.time()

# Import local file
with open("test_headline.json", "r") as f:
    dataset = json.load(f)

# Load the dataset from Hugging Face
# dataset = load_dataset("AdaptLLM/finance-tasks", "Headline")
# Extract question-answer pairs
extracted_qa_pairs = extract_qa_pairs(dataset)

# Record the end time
end_time = time.time()

# Calculate the execution time
execution_time = end_time - start_time

# Save the extracted question-answer pairs to a new JSON file
with open("processed_test_headline.json", "w") as f:
    json.dump(extracted_qa_pairs, f, indent=4)


print(f"Total number of question-answer pairs: {len(extracted_qa_pairs)}")
print(f"Execution time: {execution_time:.2f} seconds")
