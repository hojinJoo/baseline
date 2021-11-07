import csv,json,random

f = open('solution.csv', 'w', newline='')
solution_writer = csv.writer(f)

f_sample = open('sample_solution.csv', 'w', newline='')
sample_writer = csv.writer(f_sample)

with open('test/test.json','r') as json_file:
    json_data = json.load(json_file)

solution_writer.writerow(['filename','class'])
sample_writer.writerow(['filename','class'])

for i,item in enumerate(json_data['annotations']):
    solution_writer.writerow([item['file_name'],item['category_idx']])
    sample_writer.writerow([item['file_name'],random.randint(0,149)])

f.close()
f_sample.close()