import pandas as pd
import re
import csv

df = pd.DataFrame(columns=['FirstWord', 'SecondWord', 'Relation','Language'])

LANGUAGE = "it"
it = 0
counter = 0
for chunk in pd.read_csv('/content/drive/My Drive/conceptnet-assertions-5.7.0.csv.gz',chunksize = 5000, compression='gzip',error_bad_lines=False,header=None,usecols = [0,2],engine = 'python',quoting=csv.QUOTE_NONE):
  it = it+1
  print(it)
  for i in range (len(chunk)):
    #print(i)
    l = chunk.iloc[i][2].split()
    if len(l) == 6:
      rel1 = str(chunk.iloc[i][0])
      rel = rel1[7:-1]
      a3 = str(l[0])
      a1 = str(l[2])
      a2 = str(l[3])
      #print(a1)
      #print(a2)
      m1 = re.search(r'.*\/([^-]{2,3})\/.*$',a3)
      if m1:
        lang = m1.group(1)
        if a1[-2] == "/":
          m2 = re.search(r'.*\/([^-]*).*$',a1[:-2])
        else:
          m2 = re.search(r'.*\/([^-]*).*$',a1)
        word1 = m2.group(1)
        if a2[-2] == "/":
          m3 = re.search(r'.*\/([^-]*).*$',a2[:-2])
        else:
          m3 = re.search(r'.*\/([^-]*).*$',a2)
        word2 = m3.group(1)
        if lang == LANGUAGE and str(l[2][3:5]) == LANGUAGE:
          df.loc[counter] = [word1, word2, rel, lang]
          counter = counter+1
      else:
        continue
    else:
      continue

df.to_csv('CNItalianData.csv')
!cp CNItalianData.csv "drive/My Drive/"
