# -*- coding: utf-8 -*-
"""MT.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1cN95uyAIv7cd8FVJyh2LupTVThpOLrQB
"""

### mounting drive

from google.colab import drive
drive.mount('/content/drive')

### install transformers

!pip install transformers command
!pip install sentencepiece

### import necessary libraries

import os
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
import csv
from transformers import MarianMTModel, MarianTokenizer

### code from github (reading the files)

def load_task1(train_path):
  """
  Load task 1 training set and convert the tags into binary labels. 
  Paragraphs with original labels of 0 or 1 are considered to be negative examples of PCL and will have the label 0 = negative.
  Paragraphs with original labels of 2, 3 or 4 are considered to be positive examples of PCL and will have the label 1 = positive.
  It returns a pandas dataframe with paragraphs and labels.
  """
  rows=[]
  with open(os.path.join(train_path, 'results_final_final_nl.tsv')) as f:
    for line in f.readlines()[0:]:
      par_id=line.strip().split('\t')[0]
      art_id = line.strip().split('\t')[1]
      keyword=line.strip().split('\t')[2]
      country=line.strip().split('\t')[3]
      t=line.strip().split('\t')[4].lower()
      l=line.strip().split('\t')[-1]
      if l=='0' or l=='1':
        lbin=0
      else:
        lbin=1
      rows.append(
          {'par_id':par_id,
          'art_id':art_id,
          'keyword':keyword,
          'country':country,
          'text':t, 
          'label':lbin, 
          'orig_label':l
          }
        )
  df = pd.DataFrame(rows, columns=['par_id', 'art_id', 'keyword', 'country', 'text', 'label', 'orig_label']) 
  train_task1_df = df
  return train_task1_df

def load_task2(train_path, return_one_hot=True):
  # Reads the data for task 2 and present it as paragraphs with binarized labels (a list with seven positions, "activated or not (1 or 0)",
  # depending on wether the category is present in the paragraph).
  # It returns a pandas dataframe with paragraphs and list of binarized labels.
  tag2id = {
      'Unbalanced_power_relations':0,
      'Shallow_solution':1,
      'Presupposition':2,
      'Authority_voice':3,
      'Metaphors':4,
      'Compassion':5,
      'The_poorer_the_merrier':6
      }

  data = defaultdict(list)
  with open (os.path.join(train_path, 'dontpatronizeme_categories.tsv')) as f:
    for line in f.readlines()[0:]:
      par_id=line.strip().split('\t')[0]
      art_id = line.strip().split('\t')[1]
      text=line.split('\t')[2].lower()
      keyword=line.split('\t')[3]
      country=line.split('\t')[4]
      start=line.split('\t')[5]
      finish=line.split('\t')[6]
      text_span=line.split('\t')[7]
      label=line.strip().split('\t')[-2]
      num_annotators=line.strip().split('\t')[-1]
      labelid = tag2id[label]
      if not labelid in data[(par_id, art_id, text, keyword, country, start, finish, text_span, label, num_annotators)]:
        data[(par_id,art_id, text, keyword, country, start, finish, text_span,  label, num_annotators)].append(labelid)

  par_ids=[]
  art_ids=[]
  pars=[]
  keywords=[]
  countries=[]
  labels=[]
  starts = []
  finishes = []
  text_spans = []
  labels1 = []
  num_annotators1 = []

  for par_id, art_id, par, kw, co, st, fin, sp,l, num in data.keys():
    par_ids.append(par_id)
    art_ids.append(art_id)
    pars.append(par)
    keywords.append(kw)
    countries.append(co)
    starts.append(st)
    finishes.append(fin)
    text_spans.append(sp)
    labels1.append(l)
    num_annotators1.append(num)

  for label in data.values():
    labels.append(label)

  if return_one_hot:
    labels = MultiLabelBinarizer().fit_transform(labels)
  df = pd.DataFrame(list(zip(par_ids, 
                art_ids, 
                pars, 
                keywords,
                countries,
                starts, 
                finishes,
                text_spans,
                labels1,
                num_annotators1
                )), columns=['par_id',
                          'art_id', 
                          'text', 
                          'keyword',
                          'country',
                          'start',
                          'finish',
                          'text_span',
                          'label',
                          'num_annotators'
                          ])
  return df

### reading data

train_path = '/content/drive/MyDrive/MT/'
df = load_task1(train_path)

### translating task 1

def translate_sentence(sentence, language, model_name, tokenizer):
  src_text = [
      '>>' + language + '<< ' + sentence
  ]
  translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True))
  tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
  return tgt_text

# model name
# the list of models can be found here: https://huggingface.co/docs/transformers/model_doc/marian

model_name = 'Helsinki-NLP/opus-mt-en-nl'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# write all translated sentences directly to tsv file
with open('/content/drive/MyDrive/MT/dontpatronizeme_pcl_nl_5_5978.tsv', 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')

    for index, row in df.iterrows():
      sentence = translate_sentence(row['text'], 'nl', model_name, tokenizer)
      print(str(row['par_id']) + "-" + sentence[0])
      tsv_writer.writerow([row['par_id'], row['art_id'],row['keyword'],row['country'], sentence[0], row['orig_label']])

### translating task 2

def translate_sentence(sentence, language, model_name, tokenizer):
  src_text = [
      '>>' + language + '<< ' + sentence
  ]
  translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True))
  tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
  return tgt_text

model_name = 'Helsinki-NLP/opus-mt-en-ro'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# write all translated sentences directly to tsv file
with open('/content/drive/MyDrive/MT/dontpatronizeme_categories_ro.tsv', 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')

    for index, row in df.iterrows():
      sentence = translate_sentence(row['text_span'], 'ro', model_name, tokenizer)
      print(str(index) + "-" + sentence[0])
      tsv_writer.writerow([row['par_id'], row['art_id'],row['text'], row['keyword'],row['country'],row['start'], row['finish'], sentence[0], row['label'], row['num_annotators']])

### concatenate tsv files

with open('/content/drive/MyDrive/MT/results_final_final_nl.tsv', 'wt') as out_file:
  tsv_writer = csv.writer(out_file, delimiter='\t')

  for index, row in df_nl.iterrows():
    tsv_writer.writerow([row['par_id'], row['art_id'],row['keyword'],row['country'], row['text'], row['label'], row['orig_label']])
  
  for index, row in df_nl_3868.iterrows():
    tsv_writer.writerow([row['par_id'], row['art_id'],row['keyword'],row['country'], row['text'], row['label'], row['orig_label']])

### translate one single sentence for articles too big (exceding the limit)

src_text = [
    '>>es<< krishi vigyan kendras (kvks) had been instituted in every districts across india in order to fill the information gap between farmers and agricultural scientists . however their works had been found confined to laboratories only and their expertises are hardly in the knowledge domain of farmers . government should frame policies for kvks so that they should be available to farmers at least once in a week on alternative blocks . another institutional rigidity we witness is financial inaccessibility of farmers to existing financial institutions ( banks ) . in india , banks are found to release funds to manufacturing and service sectors but not on agricultural sector . too much official procedures and rigid financial securities create inaccessibility of farmers to banks . so the banks should be advised to make survey on the constructive environment wherein farmers should accede and return loans in time . another important institutional rigidity is lack of marketing institutions such as marketing farms and agents which can deal in such agricultural products . government should invite both local and nonlocal farms which can link farmers with consumers at price determined by market rate . government also should have policies such as minimum support price and construction of cold storages so that various types of agricultural products can be available during their offseason . the last most important institutional rigidities is inability of government to constitute an integral body which comprises above mentioned farmers body , financial institutes , kvks , marketing farms and government funded research institutes . if the integral body function with mutual exchange of workings with all institutional constituents , dream of doubling farmers income  can be achieved . let this article may draw attention of agricultural ministry in fulfilling its ambition ."'
]

model_name = 'Helsinki-NLP/opus-mt-en-nl'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True))
tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

print(tgt_text[0])

### write translated sentence to file

sentence = 'Dr mayengbam lalit singh onlangs eervolle pm van India gelanceerde regeling genaamd verdubbeling boeren inkomen om het welzijn van een groter deel van de indianen omhoog te tillen. de grote vragen gevoeld door mensen zijn (i) hoe zou het mogelijk zijn in korte tijd? en (ii) is het een ongrijpbare doctrine om te vegen bank van miljoenen boeren voor komende parlementsverkiezingen in 2019? over alles over india (met uitzondering van een paar staten) zo veel rigiditeiten moeten worden opgelost om het werkelijke inkomen van boeren te verhogen. deze rigiditeiten worden gekenmerkt door infrastructuur, structuren en instellingen. het huidige artikel richt zich op die rigiditeiten alleen voor manipur die achter veel Indiase staten is gebleven. wat betreft infrastructuur (inputs), de staat wordt terug gevonden onder die achterwaartse toestanden van india. Introspectie water infrastructuur, de staat heeft onder-utalising water middelen ondanks de overvloedige beschikbaarheid. voor kharif seizoen gewassen afhankelijk van regenwater sinds de staat. Mijn vorige artikel over ""channelizing mahatma gandhi national rural employment guarantee act (mgnrega) met gewas en veeverzekeringen in India "" postuleert het model over hoe het verlies te compenseren. het tweede rigide karakter van de landbouw is dat er geen structurele verandering in de loop van decennia was geweest. de landbouw in manipur richt zich op de productie van alleen ruwe granen die wordt gekenmerkt door onmiddellijke consumptie. in dit tijdperk van globalisering landbouw zou moeten hebben nauwe introspectie van vraag en aanbod model van consumenten over de hele wereld. vooral in heuvelachtige gebieden, de teelt van ruwe granen is alleen maar om te voldoen aan lokale consumptie en niet voor het verdienen van winst. tegenwoordig, boeren verklaarden de teelt van illegale gewassen zoals papaver illegaal om enorme winst te verdienen. om te voorkomen dat deze, de overheid zou moeten benadrukken op teelt van cash gewassen zoals yongchak (parkia speciosa ), kardamom, koffie, zwarte peper, enz. Krishi vigyan kendras (kvks ) was ingesteld in alle districten in India om de informatiekloof tussen landbouwers en landbouwwetenschappers te vullen. hoe hun werken waren gevonden beperkt tot laboratoria alleen en hun expertises zijn nauwelijks in het kennisgebied van landbouwers. de overheid moet beleid voor kvks zodanig dat ze beschikbaar moeten zijn voor landbouwers ten minste een keer in een week op alternatieve blokken. een andere institutionele rigiditeit die we zien is financiële ontoegankelijkheid van landbouwers aan bestaande financiële instellingen (banken). in India, banken worden gevonden om fondsen vrij te geven aan de productie- en dienstensectoren, maar niet aan de landbouwsector. een andere belangrijke institutionele rigiditeit is het ontbreken van marketingbedrijven en agenten die dergelijke landbouwproducten kunnen behandelen. zou zowel lokale als niet-lokale boeren moeten uitnodigen om consumenten te koppelen aan de door de overheid vastgestelde prijzen.'

# write the sentence to a tsv file (all tsv files will be concatenated)
with open('/content/drive/MyDrive/MT/dontpatronizeme_pcl_nl_5_7419.tsv', 'wt') as out_file:
  tsv_writer = csv.writer(out_file, delimiter='\t')
  tsv_writer.writerow(['7419', '@@23830603', 'vulnerable', 'in', sentence ,'0', '0'])