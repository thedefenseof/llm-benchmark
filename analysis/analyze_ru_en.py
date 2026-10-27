import pandas as pd

short = pd.read_csv("results/ru_en_results.csv", encoding='utf-8')

acc = short.groupby(['model', 'lang'])['correct'].mean()
acc_table = acc.unstack()

by_type = short.groupby(['model', 'lang', 'type'])['correct'].mean()

tf = short[short['type'] != 'geo']
tf_acc = tf.groupby(['model', 'lang'])['correct'].mean().unstack()

geo_acc = short[short['type'] == 'geo'].groupby(['model', 'lang'])['correct'].mean().unstack()

long = pd.read_csv("results/ru_en_long.csv", encoding='utf-8')

long['chars_per_token'] = long['chars'] / long['output_tokens']
summary = long.groupby(['model', 'lang'])[['tok_per_s', 'chars_per_token', 'cyrillic_pct']].median()

tf_acc.to_csv('ru_en_short_summary.csv')
geo_acc.to_csv('ru_en_geo_accuracy.csv')
summary.to_csv('ru_en_long_summary.csv')