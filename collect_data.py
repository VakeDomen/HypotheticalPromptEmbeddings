import csv
import re

def parse_rag_results(rag_text):
    """Extracts metrics and execution time from RAGResults string."""
    metrics_pattern = re.compile(r'\"(\w+_metrics)\": \{(.*?)\}', re.DOTALL)
    execution_time_pattern = re.compile(r'--- (\d+\.\d+) seconds ---')

    metrics_data = {}
    for match in metrics_pattern.finditer(rag_text):
        key = match.group(1)
        values = match.group(2)
        for metric in re.finditer(r'"(.*?)": (\d+\.\d+)', values):
            metrics_data[f"{key}_{metric.group(1)}"] = float(metric.group(2))
    
    execution_time_match = execution_time_pattern.search(rag_text)
    execution_time = float(execution_time_match.group(1)) if execution_time_match else None

    return metrics_data, execution_time

def collect_data():
    """Collects user input and writes to CSV."""
    csv_filename = "results.csv"
    methods = ["Naive", "HyDE", "HyPE"]

    # Define CSV headers
    headers = [
        "dataset_name", "k", "distance_metric", "method",
        "overall_precision", "overall_recall", "overall_f1",
        "retriever_claim_recall", "retriever_context_precision",
        "generator_context_utilization", "generator_noise_sensitivity_in_relevant",
        "generator_noise_sensitivity_in_irrelevant", "generator_hallucination",
        "generator_self_knowledge", "generator_faithfulness",
        # "execution_time"
    ]

    # Open the CSV file and ensure it has a header if it's new
    try:
        with open(csv_filename, 'x', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
    except FileExistsError:
        pass  # File already exists

    while True:
        print("\nEnter dataset details (or type 'exit' to finish):")
        dataset_name = input("Dataset Name: ")
        if dataset_name.lower() == 'exit':
            break
        k = input("@k (num of documents retrieved): ")
        distance_metric = input("Distance metric (cosine/euclidean): ")

        for method in methods:
            print(f"\nProcessing method: {method}")
            print("Paste the RAGResults output (type 'end' on a new line when done):")

            rag_results = []
            while True:
                line = input()
                if line.lower() == 'end':
                    break
                rag_results.append(line)

            rag_text = "\n".join(rag_results)
            metrics, execution_time = parse_rag_results(rag_text)

            # Prepare the row data
            row = [
                dataset_name, k, distance_metric, method,
                metrics.get("overall_metrics_precision", ""),
                metrics.get("overall_metrics_recall", ""),
                metrics.get("overall_metrics_f1", ""),
                metrics.get("retriever_metrics_claim_recall", ""),
                metrics.get("retriever_metrics_context_precision", ""),
                metrics.get("generator_metrics_context_utilization", ""),
                metrics.get("generator_metrics_noise_sensitivity_in_relevant", ""),
                metrics.get("generator_metrics_noise_sensitivity_in_irrelevant", ""),
                metrics.get("generator_metrics_hallucination", ""),
                metrics.get("generator_metrics_self_knowledge", ""),
                metrics.get("generator_metrics_faithfulness", ""),
                # execution_time if execution_time is not None else ""
            ]

            # Write to CSV
            with open(csv_filename, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(row)

            print("Data saved successfully!\n")

        print("\nFinished processing this dataset. Moving to next...\n")

if __name__ == "__main__":
    collect_data()
