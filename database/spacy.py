import pandas as pd
import pandas as pd
import spacy


nlp = spacy.load('en_core_web_md')


file_path = 'circRNASequence.csv'
df = pd.read_csv(file_path, header=None, names=['identifier', 'sequence'])
sequences = df['sequence'].tolist()
nucleotides = ['A', 'C', 'G', 'T']
nucleotide_features = {}


for nucleotide in nucleotides:
    doc = nlp(nucleotide)
    feature_vector = doc.vector
    nucleotide_features[nucleotide] = feature_vector


df_features = pd.DataFrame(nucleotide_features).T
df_features.columns = [f'feature_{i}' for i in range(df_features.shape[1])]
df_features.insert(0, 'Nucleotide', df_features.index)


df_features.to_csv('circRNAEmbedding_spacy.csv', index=False, header=False)
