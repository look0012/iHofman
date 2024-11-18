import re
from gensim.models import FastText
import csv
from tqdm import tqdm

csv.field_size_limit(500 * 1024 * 1024)


def read_csv(save_list, file_name):
    csv_reader = csv.reader(open(file_name))
    for row in csv_reader:
        save_list.append(row)
    return


def store_csv(data, file_name):
    with open(file_name, "w", newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(data)
    return

MyMiRBase = []
read_csv(MyMiRBase, 'circSequence.csv')

circRNACorpus = []
for row in tqdm(MyMiRBase, desc="Processing RNA sequences"):
    circRNACorpus.append(re.findall(r'.{1}', row[1]))

model = FastText(circRNACorpus, min_count=1, size=64)
circRNAEmbedding = []

for word in tqdm(list(model.wv.index2word), desc="Extracting embeddings"):
    row = [word]
    row.extend(model.wv[word])
    circRNAEmbedding.append(row)


store_csv(circRNAEmbedding, 'circ_fastext_embeddings.csv')
model.save('circRNA_model')