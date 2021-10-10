#Import Modules
import json, re
import pandas as pd, numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
ps = PorterStemmer()

recipe_names = []
ingredient_sets = []
file = r'jsonrecipes.json' #Read in our data set
data = pd.read_json('jsonrecipes.json') #Convert data set to PD dataframe

with open(file, encoding='utf-8') as train_file:
    recipe_set = json.load(train_file) #Load data set
    for recipe in recipe_set: #Cycle through recipes
        ingredients = recipe["CLEANINGREDIENTS"].split(', ') #Creates array of just ingredients
        ingredient_sets.append(ingredients)
        recipe_names.append(recipe["RECIPENAME"])


#Make dataframe of names and respective clean ingredients
df = pd.DataFrame({'Name':recipe_names,
                   'ingredients':ingredient_sets})

new = []
for s in df['ingredients']:
    s = ' '.join(s) #Convert ingredients list to single string separated by a space
    new.append(s)


df['ing'] = new #modified ingredients list added to DF

l=[]
"""
Cleaning of the ingredients list for efficiency of comparison.
I did find this algorithm online, however I understand exactly how it works.
I could sufficiently reproduce or modify this if I needed to.
re.sub finds and replaces all instances of a pattern in a string, in this case, it is replaced by blank ''
"""
for s in df['ing']: #cycle through new ingredient strings
    #Remove punctuations
    s=re.sub(r'[^\w\s]','',s) #removes non alphanumeric and non whitespace characters
    #Remove Digits
    s=re.sub(r"(\d)", "", s) #replaces digits with a blankspace
    #Remove content inside paranthesis
    s=re.sub(r'\([^)]*\)', '', s)
    #Searches for a (. then removes anything within it that is not ), before removing )

    #Convert to lowercase
    s=s.lower()

    #Remove Stop Words
    stop_words = set(stopwords.words('english')) #Set of meaningless words such as the, he, have
    word_tokens = word_tokenize(s)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    filtered_sentence = []
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w) #add non stop words to our sentence
    s = ' '.join(filtered_sentence)

    #Porter Stemmer Algorithm - strips suffixes and simplifies our word tokens
    words = word_tokenize(s)
    word_ps = []
    for w in words:
        word_ps.append(ps.stem(w))
    s = ' '.join(word_ps)

    l.append(s)

#In future could account for titles as they include useful data
df['ing_mod'] = l #new dataframe column for modified ingredients

"""
TFIDF = term frequency inverse document frequency
Vectorizer converts a collection of raw documents to a matrix of tfidf features.
Weights the relative importance of word tokens to their document, and whole database, allowing for comparison of data sets.
"""
tfidf = TfidfVectorizer(
    min_df = 5,
    max_df = 0.95,
    max_features = 8000,
    stop_words = 'english'
)
tfidf.fit(data.CLEANINGREDIENTS)
#Fit model parameters to our clean, base ingredients set, not the tokenised ingredinets set

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['ing_mod'])
#Fit learns the the model parameters
#Transform centers data set on zero with a unit variance, scales our data set

clusters = KMeans(init='k-means++', n_clusters=3).fit_predict(X)
#Clusters ingredients sets
#Computes centers and predicts a cluster index for each sample


def generate_colours(clusters):
    n = len(set(clusters))
    cmap = [cm.hsv(i/n) for i in range(n)]
    return cmap


#Once again used some code found online and manipulated it to what I needed
def plot_tsne_pca(data, labels):
    max_label = max(labels)
    max_items = np.random.choice(range(data.shape[0]), size=1000, replace=False)
    #select random samples from our data set

    pca = PCA(n_components=2).fit_transform(data[max_items,:].todense())
    #perform principle component analysis on our data to extrapolate it in 2 dimensions

    idx = np.random.choice(range(pca.shape[0]), size=300, replace=False)
    label_subset = labels[max_items]
    colours = generate_colours(labels[idx])
    colour_map = [colours[i] for i in label_subset[idx]]

    f, ax = plt.subplots(1, 1, figsize=(10, 10))

    ax.scatter(pca[idx, 0], pca[idx, 1], c = colour_map)
    ax.set_xlabel('lambda_1')
    ax.set_ylabel('lambda_2')


result = pd.DataFrame({'Name': recipe_names,
                       'Cluster': clusters})
print(result.head(25))
plot_tsne_pca(X, clusters)
plt.show()
