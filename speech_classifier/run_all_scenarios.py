import subprocess
import csv


target_groups = ['black', 'women', 'lgbt']
output_file = '../output/full_table.csv'

def read_metrics(filename, data_name, target_group):
    metrics = {}
    with open(filename, 'r') as file:
        for line in file:
            metrics['Scenario'] = data_name
            metrics['Target Group'] = target_group
            parts = line.split('=')
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip()
                try:
                    metrics[key] = float(value)
                except ValueError:
                    print(f"Could not convert {value} to float.")
    return metrics

def write_to_csv(output_filename, data):
    headers = data[0].keys()
    with open(output_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        for d in data:
            writer.writerow(d)

if __name__ == '__main__':
    all_metrics = []

    for data_name in data_names:
        # Command to run the speech_classifier.py
        command = ["python", "speech_classifier.py", data_name]
        subprocess.run(command)
    
    for data_name in data_names:
        for target_group in target_groups:
            # Now collect the results from speech classifier and saved into a csv file
            results_file = f'../output/results_{data_name}/scenario_{data_name}_data_{target_group}.csv_(2_classes).txt'
            metrics = read_metrics(results_file, data_name, target_group)
            all_metrics.append(metrics)
        
    write_to_csv(output_file, all_metrics)