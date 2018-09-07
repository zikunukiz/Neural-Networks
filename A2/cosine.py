import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle

def cos(n1, n2):
    embeddings_file = 'data/word_embeddings.p'
    embed_dict = load_embeddings(embeddings_file)
    
    vocabulary = np.array(embed_dict.keys())
    word_vec = np.array(embed_dict.values())
    
    w1 = word_vec[[n1], :]
    w2 = word_vec[[n2], :]
    s1 = vocabulary[[n1]]
    s2 = vocabulary[[n2]]

    value = np.dot(w1, w2.T)[0]/(np.sqrt(np.dot(w1, w1.T)[0])*np.sqrt(np.dot(w2, w2.T)[0]))
    print s1[0]+' & '+s2[0]
    print value[0]

def show():
    # printing out words to see their indices in the dictionary
    embeddings_file = 'data/word_embeddings.p'
    embed_dict = load_embeddings(embeddings_file)
    i = -1
    for key in embed_dict.keys():
        i = i + 1
        print str(i) + ' ' + key

#(A, -8.91121547105)
#(man, -9.31004987916)

def plot():
    
    bleu1 = np.array([ 3.5,  4.3,  7.3,  6. ])
    x1 = np.array([  5000.,  10000.,  15000.,  20000.])
    
    bleu2 = np.array([ 0.5,  2.7,  6.8,  5.9])
    x2 = np.array([  5000.,  10000.,  15000.,  20000.])
    
    bleu3 = np.array([ 0. ,  6.4,  6.7,  5.6])
    x3 = np.array([  5000.,  10000.,  15000.,  20000.])

    bleu4 = np.array([ 0.3,  2.1,  7. ,  6.3])
    x4 = np.array([  5000.,  10000.,  15000.,  20000.])
    
    bleu5 = np.array([3.7,  3.4,  7.6,  6.3])
    x5 = np.array([ 5000.,  10000.,  15000.,  20000.])
    
    bleu6 = np.array([0.5,  2.8,  6.6,  6.2])
    x6 = np.array([ 5000.,  10000.,  15000.,  20000.])
    
    bleu7 = np.array([0. ,  3.8,  8. ,  6.9])
    x7 = np.array([ 5000.,  10000.,  15000.,  20000.])
    
    bleu8 = np.array([ 3.7,  3.9,  6.8,  6.9])
    x8 = np.array([ 5000.,  10000.,  15000.,  20000.])
    
    bleu9 = np.array([ 0. ,  3.7,  8.1,  6.6])
    x9 = np.array([ 5000.,  10000.,  15000.,  20000.])
    
    bleu10 = np.array([ 0. ,  3.9,  7.9,  6. ])
    x10 = np.array([ 5000.,  10000.,  15000.,  20000.])
    
    
    plt.plot(x1, bleu1, marker='o', linestyle='--', color='magenta', label='0.43, 0.23')
    plt.plot(x2, bleu2, marker='o', linestyle='--', color='blue', label='0.1, 0.23')
    plt.plot(x3, bleu3, marker='o', linestyle='--', color='green', label = "0.43, 0.9")
    plt.plot(x4, bleu4, marker='o', linestyle='--', color='black', label = "0.1, 0.9")
    plt.plot(x5, bleu5, marker='o', linestyle='--', color='yellow', label = "0.43, 0.1")
    plt.plot(x6, bleu6, marker='o', linestyle='--', color='orange', label = "0.6, 0.1")
    plt.plot(x7, bleu7, marker='o', linestyle='--', color='purple', label = "0.5, 0.1")
    plt.plot(x8, bleu8, marker='o', linestyle='--', color='pink', label = "0.5, 0.15")
    plt.plot(x9, bleu9, marker='o', linestyle='--', color='red', label = "0.5, 0.05")
    plt.plot(x10, bleu10, marker='o', linestyle='--', color='cyan', label = "0.5, 0.03")
    
    plt.xlim((0, 25000))
    plt.ylim((0, 10))
    plt.xlabel('Points')
    plt.ylabel('BLEU Scores')
    plt.title('BLEU Scores vs data points for first epoch')
    plt.legend(loc='lower right')
    


# 0.43, 0.9
#>>> mlbl.B
#array([ 2.4,  5.6,  5.9,  6.8,  6.8])
#>>> mlbl.x
#array([  5000.,  10000.,  25000.,  40000.,  45000.])
#55.24 mins, 7.0

#################################################################################

# CONSOLE 25
#    d['learning_rate'] = 0.43
#    d['momentum'] = 0.23
#>>> mlbl.B
#array([ 3.5,  4.3,  7.3,  6. ,  3.8,  5. ,  4.8,  6.6,  5.8,  3.5,  6.5,
#        5.5,  5.8])
#>>> mlbl.x
#array([  5000.,  10000.,  15000.,  20000.,  25000.,  30000.,  35000.,
#        40000.,  45000.,  50000.,  55000.,  60000.,   5000.])
# BLEU score = 6.6



#    d['learning_rate'] = 0.1
#    d['momentum'] = 0.23
#array([ 0.5,  2.7,  6.8,  5.9,  3.3,  3.9,  4.4,  6.4,  5. ,  5.2,  5.5,
#        5.7,  5.7])
#>>> mlbl.x
#array([  5000.,  10000.,  15000.,  20000.,  25000.,  30000.,  35000.,
#        40000.,  45000.,  50000.,  55000.,  60000.,   5000.])
#BLEU score = 6.5


#    d['learning_rate'] = 0.43
#    d['momentum'] = 0.9
#>>> mlbl.B
#array([ 0. ,  6.4,  6.7,  5.6,  5.1,  4.5,  3.7,  6.1,  5.5,  3.6,  5.1,
#        4.9,  4.7])
#>>> mlbl.x
#array([  5000.,  10000.,  15000.,  20000.,  25000.,  30000.,  35000.,
#        40000.,  45000.,  50000.,  55000.,  60000.,   5000.])
#BLEU score = 6.1

#    d['learning_rate'] = 0.1
#    d['momentum'] = 0.9
#>>> mlbl.B
#array([ 0.3,  2.1,  7. ,  6.3,  6.5,  4.2,  4.3,  5.4,  4.6,  5.4,  5.1,
#        5.2,  6.4])
#>>> mlbl.x
#array([  5000.,  10000.,  15000.,  20000.,  25000.,  30000.,  35000.,
#        40000.,  45000.,  50000.,  55000.,  60000.,   5000.])
# BLEU score = 6.0


####################################################################################
# CONSOLE 30
#    d['learning_rate'] = 0.43
#    d['momentum'] = 0.1
#>>> mlbl.B
#array([ 3.7,  3.4,  7.6,  6.3,  3.1,  5.5,  4.4,  6.8,  7. ,  3.1])
#>>> mlbl.x
#array([  5000.,  10000.,  15000.,  20000.,  25000.,  30000.,  35000.,
#        40000.,  45000.,  50000.])
#BLEU score = 6.8

#    d['learning_rate'] = 0.1
#    d['momentum'] = 0.1
#>>> mlbl.B
#array([ 0.5,  2.8,  6.6,  6.2,  0. ])
#>>> mlbl.x
#array([  5000.,  10000.,  15000.,  20000.,  25000.])

#    d['learning_rate'] = 0.6
#    d['momentum'] = 0.1
#>>> mlbl.B
#array([ 0. ,  4.6,  7.3,  5.9])
#>>> mlbl.x
#array([  5000.,  10000.,  15000.,  20000.])
#BLEU score = 6.8

#    d['learning_rate'] = 0.5
#    d['momentum'] = 0.1
#>>> mlbl.B
#array([ 0. ,  3.8,  8. ,  6.9])
#>>> mlbl.x
#array([  5000.,  10000.,  15000.,  20000.])
#BLEU score = 7.0
#TIME 15.48

#    d['learning_rate'] = 0.5
#    d['momentum'] = 0.15
#>>> >>> mlbl.B
#array([ 3.7,  3.9,  6.8,  6.9,  5.5])
#>>> mlbl.x
#array([  5000.,  10000.,  15000.,  20000.,  25000.])

#    d['learning_rate'] = 0.5
#    d['momentum'] = 0.12
#>>> mlbl.B
#array([ 0. ,  4.2,  7.1])
#>>> mlbl.x
#array([  5000.,  10000.,  15000.])

# CONSOLE 35
# initially want learning rate to dominate (coefficient infront of the gradient signal)
# but learning rate not too large, otherwise overshoot
#    d['learning_rate'] = 0.5
#    d['momentum'] = 0.05
#>>> mlbl.x
#array([  5000.,  10000.,  15000.,  20000.,  25000.,  30000.,  35000.,
#        40000.,  45000.,  50000.,  55000.,  60000.,   5000.])
#>>> mlbl.B
#array([ 0. ,  3.7,  8.1,  6.6,  5.3,  5.7,  4. ,  6.7,  7.4,  3.7,  6.4,
#        5.3,  6.5])

#    d['learning_rate'] = 0.5
#    d['momentum'] = 0.01
#>>> mlbl.B
#array([ 0. ,  3.6,  7.8,  6.4])
#>>> mlbl.x
#array([  5000.,  10000.,  15000.,  20000.])

#    d['learning_rate'] = 0.5
#    d['momentum'] = 0.03

#Use of uninitialized value in division (/) at ./multi-bleu.perl line 123, <STDIN> line 250.

#>>> mlbl.B
#array([ 0. ,  3.9,  7.9,  6. ])
#>>> mlbl.x
#array([  5000.,  10000.,  15000.,  20000.])



def load_embeddings(file_name):
    """ Load in the embeddings """
    return pickle.load(open(file_name, 'rb'))
    
    
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
    
    # 244 elephant 268 tomato 391 umbrella 422 cheese 471 flowers 484 computer 
    # 565 bedroom 718 racquets 719 laptops 720 photos 739 bathtub 785 advertisement 849 traffic
    # 875 Nintendo 847 game 677 games

    
#>>> cos(1057,1055)
#vehicles & pedestrians
#0.474547161291
#>>> cos(1038, 321)
#football & soccer
#0.737345239006
#>>> cos(842,60)
#big & small
#0.745515376114
#>>> cos(842,760)
#big & large
#0.748236941148
#>>> cos(878,1)
#motorcyclists & all
#0.120465130251
#>>> cos(878,594)
#motorcyclists & watermelon
#-0.00990755672344
#>>> cos(878,747)
#motorcyclists & sleeping
#0.303559425376
#>>> cos(878,255)
#motorcyclists & farm
#0.126654449068
#>>> cos(878,1057)
#motorcyclists & vehicles
#0.371195102169
#motorcyclists & pedestrians
#0.402447608851
#>>> cos(318,483)
#orange & banana
#0.493180895151
#purple & watermelon
#0.128966612018
#>>> cos(1061, 1)
#purple & all
#0.282730532702
#>>> cos(838, 1038)
#sandwiches & football
#0.388551316547
#>>> cos(113, 379)
#gathered & surrounded
#0.462729388673
#>>> cos(1045, 716)
#decorated & painted
#0.538570468426
#>>> cos(419, 320)
#onto & mounted
#0.457314607103
#>>> cos(875, 847)
#Nintendo & game
#0.199276110092
#>>> cos(875, 677)
#Nintendo & games
#0.21000140703
#>>> cos(875, 244)
#Nintendo & elephant
#0.241157455304
#>>> cos(484, 244)
#computer & elephant
#0.401478326036
#>>> cos(878, 244)
#motorcyclists & elephant
#0.0887606891413
#>>> cos(878, 719)
#motorcyclists & laptops
#0.111943095345
#>>> cos(878, 849)
#motorcyclists & trafficc
#0.297488293496
#>>> cos(878, 471)
#motorcyclists & flowers
#0.419958890243
#>>> cos(878, 391)
#motorcyclists & umbrella
#-0.00601722912845
#>>> cos(878, 268)
#motorcyclists & tomato
#0.0892285355168
