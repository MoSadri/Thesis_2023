import csv

# Typically I use data_name to signify which data I used, like 'black' means the file black_dataset.csv, 'balanced' means the file balanced_dataset.csv
#data_name = 'black' 
#data_name = 'women'
data_name = 'balanced'

def count_groups(csv_file, output_file):

    # These are the columns of targeted groups
    column_start = 2
    column_last = 8
    total_row_count = 0
    
    target_groups = {}

    default_char = '?'
    with open(csv_file, 'r', encoding='utf-8', errors='replace') as file:
        reader = csv.DictReader(file)
        column_names = reader.fieldnames

        # print(column_names[column_start:column_last+1])

        # Initialize the dictionary
        for i in range(column_start, column_last+1):
            target_groups[column_names[i]] = 0

        for row in reader:
            for i in range(column_start, column_last+1):
                if row[column_names[i]] == 'True' or row[column_names[i]] == 'TRUE':
                    target_groups[column_names[i]] += 1
            total_row_count += 1

    print(f'Writing to file {output_file}')

    with open(output_file, 'w') as file:
        file.write(f'Total rows = {total_row_count}\n')
        for target_group in target_groups:
            group_percentage = target_groups[target_group] / total_row_count * 100
            file.write(f'{target_group} count = {target_groups[target_group]}, percentage = {group_percentage}%\n')

if __name__ == "__main__":
    csv_file_path = f"../data/{data_name}_dataset.csv"  # Replace with the actual path to your CSV file
    output_file_path = f"../output/{data_name}_groups_counts.txt"
    count = count_groups(csv_file_path, output_file_path)
