import lm_tools, coco_proc
import numpy as np

if __name__ == '__main__':

    loc = 'models/mlbl_model.pkl'
    beam_width = 1
    z, zd, zt = coco_proc.process(context=5)

    net = lm_tools.load_model(loc)
    net.batchsize = 1

    # use the 1st example file: COCO_val2014_000000183391.jpg    
    VIM_example = zd['IM'][22, :]
    
    lm_tools.search(net, z, im=VIM_example, k=beam_width, disp_word=True)
    