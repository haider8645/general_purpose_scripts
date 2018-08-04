import itertools

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import euclidean_distances
from sklearn.svm import SVC
from sklearn.preprocessing import LabelBinarizer
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import classification_report,confusion_matrix
import sys
sys.path.insert(0,'/home/haider/caffe/python-scripts/mnist')
import tsne
import metrics as met

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('without normalization')

def normalizeIris(iris):
        iris[:,:4] = iris[:,:4]-iris[:,:4].mean(axis = 0)
        imax = iris.max(axis=0)
        iris[:,:4] = iris[:,:4]/imax[:4]
        return iris

def MLP(X,y):
    from sklearn.neural_network import MLPClassifier
 #   X = normalizeIris(x_train)
    X= x_train
    y= y_train

    print target_names[0]
    X_class1= X[0:49]
    print 'mean: ' , X_class1.mean(axis=0)
    X_class2= X[50:99]
    print target_names[1]
    print 'mean: ' , X_class2.mean( axis =0)
    X_class3= X[100:149]
    print target_names[2]
    print 'mean: ' , X_class3.mean(axis =0)

    mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(3), random_state=1)
    mlp.fit(X, y)
    print mlp.score(X,y)
    predictions = mlp.predict(X)
    cnf_matrix = confusion_matrix(y_train,predictions)
    print(classification_report(y_train,predictions))
    plot_confusion_matrix(cnf_matrix, classes=target_names,
                      title='Confusion matrix, without normalization')

def LDA_():
    #LDA
    fig,ax = plt.subplots()
    x_train_lda = LDA(n_components=2).fit(x_train,y_train_img).transform(x_test)
    cax=plt.scatter(x_train_lda[:, 0], x_train_lda[:, 1], 20, labels, edgecolors='face',alpha=1,cmap=plt.cm.get_cmap('jet', N))
    cbar=plt.colorbar(ticks=tick)
    plt.clim(-0.5,N-0.5)
    cbar.ax.set_yticklabels(target_names)  # vertically$
    plt.title (title)
    plt.xlabel('principal component - 1')
    plt.ylabel('principal component - 2')
    fig.tight_layout()
    plt.savefig(title)


def CCA_():
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(labels)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    #CCA
    x_train_cca = CCA(n_components=2).fit(x_train, onehot_encoded).transform(x_train)
    #ax1.scatter(x_train_cca[:, 0], x_train_cca[:, 1], s=20, c=y_train, edgecolors='face')
    #ax1.set_title ('CCA on Iris dataset')
    #ax1.set_xlabel('dimension - 1')
    #ax1.set_ylabel('dimension - 2')

def PCA_(y_train,title):
    #PCA with scaling
    fig,ax = plt.subplots()
    x_train_pca = PCA(n_components=2).fit(x_train).transform(x_train)
    cax=plt.scatter(x_train_pca[:, 0], x_train_pca[:, 1], 20, y_train, edgecolors='face',alpha=1,cmap=plt.cm.get_cmap('jet', N))
    cbar=plt.colorbar(ticks=tick)
    plt.clim(-0.5,N-0.5)
    cbar.ax.set_yticklabels(target_names)  # vertically$
    plt.title (title)
    plt.xlabel('principal component - 1')
    plt.ylabel('principal component - 2')
    fig.tight_layout()
    plt.savefig(title)


def tsne_(y_train,title,pred,title_second):
    fig,ax = plt.subplots()
    Y = tsne.tsne(x_train, no_dims= 2, initial_dims=784, perplexity=30.0)
    cax=plt.scatter(Y[:, 0], Y[:, 1], 20, y_train, edgecolors='face',alpha=1,cmap=plt.cm.get_cmap('jet', N))
    cbar=plt.colorbar(ticks=tick)
    plt.clim(-0.5,N-0.5)
    cbar.ax.set_yticklabels(target_names)  # vertically$
    plt.title (title)
    plt.xlabel('t-SNE dimension - 1')
    plt.ylabel('t-SNE dimension - 2')
    fig.tight_layout()
    plt.savefig(title)

    fig,ax = plt.subplots()
    cax=plt.scatter(Y[:, 0], Y[:, 1], 20, pred, edgecolors='face',alpha=1,cmap=plt.cm.get_cmap('jet', N))
    cbar=plt.colorbar(ticks=tick)
    plt.clim(-0.5,N-0.5)
    cbar.ax.set_yticklabels(target_names)  # vertically$
    plt.title (title_second)
    plt.xlabel('t-SNE dimension - 1')
    plt.ylabel('t-SNE dimension - 2')
    fig.tight_layout()
    plt.savefig(title_second)



def kmeans_():

# use features for clustering
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=N, init = 'k-means++')
    #features = np.reshape(x_train, newshape=(features.shape[0], -1))
    km_trans = km.fit_transform(x_train)
    pred = km.predict(x_train)
    print pred.shape
    print('acc=', met.acc(y_train, pred), 'nmi=', met.nmi(y_train, pred), 'ari=', met.ari(y_train, pred))
    return km_trans,pred

if __name__ == "__main__":

    dataset = sys.argv[1]

    if dataset == 'mnist':                
        target_names = ['zero','one','two','three','four','five','six','seven','eight','nine']

        from keras.datasets import mnist

        (x_train_img, y_train_img), (x_test, y_test) = mnist.load_data()

        x_train = x_train_img * 0.00392157
        x_test  = x_test * 0.00392157


        x_train = x_train[:1000]
        y_train  = y_train_img[:1000]

        x_train = x_train.reshape(1000,784)
 


        N=10

        title = 'K-means Clusters with True Labels of MNIST'
        title_second = 'Kmeans Predicted Labels of MNIST'
        tick = [0,1,2,3,4,5,6,7,8,9]

    elif dataset == 'iris':
        from sklearn import datasets
        iris = datasets.load_iris() 
        target_names = iris.target_names
        x_train = iris.data
        y_train = iris.target
        labels = y_train
        N=3
        tick=[0,1,2]
        title = 'K-means Clusters with True Labels Iris'    
        title_second='K-means Predicted Labels of Iris'       
    x_train,pred = kmeans_()
    tsne_(y_train,title,pred,title_second)
