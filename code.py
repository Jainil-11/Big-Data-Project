import pandas as pd
import numpy as np
import seaborn as sns
from mlxtend.frequent_patterns import apriori , association_rules
from collections import defaultdict
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 999)
pd.set_option('display.max_rows', 999)
plt.style.use('fivethirtyeight')

