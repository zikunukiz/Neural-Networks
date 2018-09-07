"""
Dataset configuration
"""

#-----------------------------------------------------------------------------#
# Paths to MSCOCO
#-----------------------------------------------------------------------------#
paths = dict()

# JSON annotations
paths['sentences_coco_train'] = 'data/sentences_coco_train.json'
paths['sentences_coco_val'] = 'data/sentences_coco_val.json'

# VGG19 features
paths['train'] = 'data/train.npy'
paths['dev'] = 'data/val.npy'
paths['test'] = 'data/test.npy'

# Data splits
paths['coco_train'] = 'data/train_img_id.txt'
paths['coco_val'] = 'data/val_img_id.txt'
paths['coco_test'] = 'data/test_img_id.txt'

# Word embeddings
paths['embedding'] = 'data/word_embeddings.p'

# Visualization examples
paths['coco_val_example'] = [
    '../data/example_val_imgs/COCO_val2014_000000183391.jpg',
    '../data/example_val_imgs/COCO_val2014_000000535934.jpg'    
    ]
