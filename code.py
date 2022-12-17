import pandas as pd
import numpy as np
import seaborn as sns
from mlxtend.frequent_patterns import apriori , association_rules
from collections import defaultdict
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 999)
pd.set_option('display.max_rows', 999)
plt.style.use('fivethirtyeight')

from google.colab import drive
drive.mount('/content/drive')

file="/content/drive/MyDrive/Main With Header.csv"
a1= pd.read_csv(file)

l=len(a1['White_king_c'])
for i in range(l):
  a1.loc[i,'Wk']="WK"+a1.loc[i,"White_king_c"]+str(a1.loc[i,"White_king_r"])
  
for i in range(l):
  a1.loc[i,'Wr']="WR"+a1.loc[i,"White_rook_c"]+str(a1.loc[i,"White_rook_r"])
  a1.loc[i,'Bk']="BK"+a1.loc[i,"Black_king_c"]+str(a1.loc[i,"Black_king_r"])
  
a2=a1.loc[:,['Wk','Wr','Bk','Output']].copy()
a2.head()

a2.to_csv( "/content/drive/MyDrive/Final Data Set.csv", index=False, encoding='utf-8-sig')

file_b="/content/drive/MyDrive/Final Data Set.csv"
df = pd.read_csv(file_b)

l=df.Wk.unique()
for i in l:
  df[i] = np.where(df['Wk']==i,1,0)

l=df.Wr.unique()
for i in l:
  df[i] = np.where(df['Wr']==i,1,0)
  
l=df.Bk.unique()
for i in l:
  df[i] = np.where(df['Bk']==i,1,0)

l=df.Output.unique()
for i in l:
  df[i] = np.where(df['Output']==i,1,0)

df.drop('Wk', inplace=True, axis=1)
df.drop('Wr', inplace=True, axis=1)
df.drop('Bk', inplace=True, axis=1)
df.drop('Output', inplace=True, axis=1)
df.head()

df.to_csv( "/content/drive/MyDrive/Binary Data Set.csv", index=False, encoding='utf-8-sig')

import time
df=pd.read_csv("/content/drive/MyDrive/Binary Data Set.csv")
Time=[]
for i in range(6,10):
  start=time.time()
  frq_items1 = apriori(df,min_support=0.1*i, use_colnames = True)
  end=time.time()
  Time.append(end-start)
  print("Minsup=",0.1*i," Time:",end-start)
  
!pip install --quiet pyspark

from pyspark import SparkContext
sc = SparkContext("local" , "Apriori")

file=pd.read_csv("/content/drive/MyDrive/Final Data Set.csv")
file.head()

file =  sc.textFile("/content/drive/MyDrive/Final Data Set.csv")
df1=file.map(lambda line: line.split(','))
temp1=df1.collect()
temp = sc.broadcast(temp1)
df2=file.flatMap(lambda l: l.split(','))
df3=df2.map(lambda l: (l,1))
df4=df3.reduceByKey(lambda a,b: a+b)
print(df4.collect())
Time2=[]
minsup=28056*0.9
df5=df4.filter(lambda l: l[1]>=minsup)
resultrdd=df5.map(lambda l: (l[0],l[1]))
f1=df5.map(lambda l: l[0])
F1=f1.distinct()


def candidate_gen(record):
  if(type(record[0])==tuple):
    l1=list(record[0])
    l2=record[1]
  else:
    l1=[record[0]]
    l2=record[1]
  
  flag=False
  for i in l1:
    if(i==l2):
      flag=True
      break

  if(flag==False):
    l1.append(l2)

  
  l1.sort()
  l1=tuple(l1)
  return l1

def counting(l):
  count=0
  for i in temp.value:
    flag2=True
    for j in range(len(l)):
      flag=False
      for k in range(len(i)):
        if(i[k]==l[j]):
          flag=True
          break
      if(flag==False):
        flag2=False
        break
    if(flag2==True):
      count=count+1
  return count

import time
a=2
start_time=time.time()
while(f1.isEmpty()==False):
  candidate=f1.cartesian(F1)
  candidate=candidate.map(lambda l: candidate_gen(l))
  candidate=candidate.filter(lambda l: len(l)==a)
  candidate=candidate.distinct()
  support_count=candidate.map(lambda l: (l,counting(l)))
  support_count=support_count.filter(lambda l: l[1]>=minsup)
  f1=support_count.map(lambda l: l[0])
  #print("Length ",a," frequent itemsets")
  a=a+1
  resultrdd = resultrdd.union(f1)
end_time=time.time()
print(end_time-start_time)
Time2.append(end_time-start_time)

print(Time)
print(Time2)

import numpy as np
import matplotlib.pyplot as plt
# set width of bar
barWidth = 0.25
fig = plt.subplots(figsize =(12, 8))

br1 = np.arange(len(Time))
br2 = [x + barWidth for x in br1]

plt.bar(br1, Time, color ='r', width = barWidth,edgecolor ='grey', label ='Sequential')
plt.bar(br2, Time2, color ='g', width = barWidth,edgecolor ='grey', label ='YAFIM')

plt.xlabel('Minsup')
plt.ylabel('Time')
plt.xticks([r + barWidth for r in range(len(Time))],['0.6', '0.7', '0.8', '0.9'])
plt.legend()
plt.show()

aa=pd.read_csv("/content/drive/MyDrive/Binary Data Set.csv")
Timen1=[]

for i in range(4,9):
  ab=aa.head(1000*i)
  start=time.time()
  frq_items1 = apriori(ab,min_support=0.6, use_colnames = True)
  end=time.time()
  Timen1.append(end-start)
  print("Rows=",1000*i," Time:",end-start)
  
Timen2=[]
minsup=0.6*8000
ab=aa.head(8000)
ab.to_csv("/content/drive/MyDrive/New Data Set.csv", index=False, encoding='utf-8-sig')
file = sc.textFile("/content/drive/MyDrive/New Data Set.csv")
df1=file.map(lambda line: line.split(','))
df2=file.flatMap(lambda l: l.split(','))
df3=df2.map(lambda l: (l,1))
df4=df3.reduceByKey(lambda a,b: a+b)
df5=df4.filter(lambda l: l[1]>=minsup)
resultrdd=df5.map(lambda l: (l[0],l[1]))
f1=df5.map(lambda l: l[0])
F1=f1.distinct()

a=2
start_time=time.time()
candidate=f1.cartesian(F1)
candidate=candidate.map(lambda l: candidate_gen(l))
candidate=candidate.filter(lambda l: len(l)==a)
candidate=candidate.distinct()
support_count=candidate.map(lambda l: (l,counting(l)))
support_count=support_count.filter(lambda l: l[1]>=minsup)
f1=support_count.map(lambda l: l[0])
#print("Length ",a," frequent itemsets")
resultrdd = resultrdd.union(f1)
end_time=time.time()
print(end_time-start_time)
Timen2.append(end_time-start_time)

print(Timen1)
print(Timen2)

import numpy as np
import matplotlib.pyplot as plt
# set width of bar
barWidth = 0.25
fig = plt.subplots(figsize =(12, 8))

br1 = np.arange(len(Timen1))
br2 = [x + barWidth for x in br1]

plt.bar(br1, Timen1, color ='r', width = barWidth,edgecolor ='grey', label ='Sequential')
plt.bar(br2, Timen2, color ='g', width = barWidth,edgecolor ='grey', label ='YAFIM')

plt.xlabel('Rows')
plt.ylabel('Time')
plt.xticks([r + barWidth for r in range(len(Timen1))],['4000', '5000', '6000', '7000','8000'])
plt.legend()
plt.show()
