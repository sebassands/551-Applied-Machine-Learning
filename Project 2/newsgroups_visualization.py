import numpy as np
from sklearn.datasets import fetch_20newsgroups
import seaborn as sb
import matplotlib.pyplot as plt 



twenty_train = fetch_20newsgroups(subset='train', remove=(['headers', 'footers', 'quotes']))

sb.set(color_codes=True)
plt.figure(figsize=(20,15))
sb.distplot(twenty_train.target, bins=np.arange(twenty_train.target.min(), twenty_train.target.max() + 2))
plt.xlabel('Newsgroup', size=25)
plt.ylabel('Distribution', size=25)
plt.title('Distribution of Newsgroup Topics', size=30)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()
