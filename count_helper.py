import pandas as pd


samples_df = pd.read_csv('results/samples.csv')
class_names = samples_df['class_name'].unique()

task_df = samples_df[samples_df['sample_type'] == 'task']
task_names = task_df['name'].unique()

samples_df = samples_df[samples_df['sample_type'] == 'example']

categories = {'TP': 6, 'FN': 3, 'FP': 3}

word_counts = {class_name: 0 for class_name in class_names}

for task_name in task_names:
    for category, counter in categories.items():
        for i in range(1, counter + 1):
            name = f'{task_name} {category} Example {i}'
            example = samples_df[samples_df['name'] == name]
            class_name = example['class_name'].iloc[0]
            original_sentence = example['original_sentence'].iloc[0]
            
            word_counts[class_name] += len(original_sentence.split())

task_count = len(task_names) / len(class_names)

# get total word count for each class
print(f'Total word counts per class: {word_counts}')
# get average word count for each class
average_word_count = {class_name: int(count / task_count) for class_name, count in word_counts.items()}
print(f'Average word counts per class: {average_word_count}')
