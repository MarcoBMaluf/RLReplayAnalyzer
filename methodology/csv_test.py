import os
import pandas as pd

formatted_data_dir = '/Users/marcomaluf/Desktop/Unfinished Projects/paper/formatted_data'
formatted_data = os.listdir(formatted_data_dir)
len(formatted_data)
formatted_data[0]

test = pd.read_csv(f'{formatted_data_dir}/{formatted_data[0]}').drop(columns=['Unnamed: 0'])
test
test.info()

raw_test = pd.read_csv('/Users/marcomaluf/Desktop/Unfinished Projects/paper/raw CSVs/0A9699064817DA31F036DCA0575AA950.csv').drop(columns=['Unnamed: 0'])
raw_test.info(verbose=True)

new_test = pd.read_csv('/Users/marcomaluf/Desktop/Unfinished Projects/paper/formatted_data/OCE P-G BOW vs WC G2 2026-01-18.20.01.csv')
last_test = pd.read_csv('/Users/marcomaluf/Desktop/Unfinished Projects/paper/preprocessed_data/APAC P-F SE vs Star G1 2026-01-18.14.41.csv.csv')