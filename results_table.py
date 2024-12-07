import os
import json
import pandas as pd
json_files = [f for f in os.listdir('.') if f.startswith('evaluation_results') and f.endswith('.json')]
data = []
for json_file in json_files:
    with open(json_file, 'r') as file:
        metrics = json.load(file)
        metrics['Model'] = json_file.replace("evaluation_results_","") 
        data.append(metrics)
df = pd.DataFrame(data)
df = df[['Model', 'Precision', 'Recall', 'F1 Score', 'Accuracy']]
def highlight_best(s):
    is_best = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_best]
styled_df = df.style.apply(highlight_best, subset=['Precision', 'Recall', 'F1 Score', 'Accuracy'])
styled_df.to_excel('comparison_table.xlsx', index=False)
print(df)