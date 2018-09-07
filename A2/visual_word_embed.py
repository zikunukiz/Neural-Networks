import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
 
def main():    
    embeddings_file = 'data/word_embeddings.p'
    embed_dict = load_embeddings(embeddings_file)
    
    vocabulary = np.array(embed_dict.keys())
    word_vec = np.array(embed_dict.values())

    ############################################################################
    # You should modify this part by selecting a subset of word embeddings 
    # for better visualization
    ############################################################################
    
#    word_vec = word_vec[[1057, 1055, 878, 747, 653, 661, 824, 862, 1061, 616, 579, 853, 838, 662], :]    
#    vocabulary = vocabulary[[1057, 1055, 878, 747, 653, 661, 824, 862, 1061, 616, 579, 853, 838, 662]]
    
#    word_vec = word_vec[[594, 146, 318, 483, 783, 255, 480]+ [113, 624, 107, 319, 985, 1020], :]    
#    vocabulary = vocabulary[[594, 146, 318, 483, 783, 255, 480]+ [113, 624, 107, 319, 985, 1020]]

    
#    word_vec = word_vec[[419, 643, 984, 637]+[1038, 950, 321, 120, 743] + [113, 379, 624, 107, 319, 540, 320, 1045, 985, 716, 1020], :]    
#    vocabulary = vocabulary[[419, 643, 984, 637]+[1038, 950, 321, 120, 743] + [113, 379, 624, 107, 319, 540, 320, 1045, 985, 716, 1020]]
#    
#    word_vec = word_vec[:,:]
#    vocabulary = vocabulary[:]

    # [419, 643, 984, 637]
    # 419 onto 643 across 984 inside 637 outside
    
    # [842, 230, 760, 60, 952, 102]
    # 842 big 230 huge 760 large 60 small 952 long 102 new
    
    # [1, 518, 884, 341, 172, 169, 699, 9]
    # 1 all 518 many 884 some 341 several 172 few 169 two 699 three 9 four
    
    # [113, 379, 624, 107, 319, 540, 320, 1045, 985, 716, 1020]
    # 113 gathered 379 surrounded 624 seated 107 filled 319 covered 540 lined 
    # 320 mounted 1045 decorated 985 attached 716 painted 1020 dressed
    
    # [1057, 1055, 878, 853, 838, 662]
    # 1057 vehicles 1055 pedestrians 878 motorcyclists 853 doughnut 838 sandwiches 662 kitchen
    
    # 747 sleeping 653 kicking 661 boxing 824 dining 862 performing
    # 1061 purple 616 green 579 black
    
    # [1038, 950, 321, 120, 743]
    # 1038 football 950 tennis 321 soccer 120 baseball 743 Tennis
    
    # [594, 146, 318, 483, 783, 255, 480]
    # 594 watermelon 146 apple 318 orange 483 banana 783 fruit 255 farm 480 tree

    word_vec = word_vec[[1035, 676, 878, 375, 1055, 594, 391, 268, 422],:]
    vocabulary = vocabulary[[1035, 676, 878, 375, 1055, 594, 391, 268, 422]]

    ############################################################################
    
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)    
    Y = tsne.fit_transform(word_vec)
 
    plt.scatter(Y[:, 0], Y[:, 1])
    for label, x, y in zip(vocabulary, Y[:, 0], Y[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.show()

    
def load_embeddings(file_name):
    """ Load in the embeddings """
    return pickle.load(open(file_name, 'rb'))
    

if __name__ == '__main__':    
    main()
