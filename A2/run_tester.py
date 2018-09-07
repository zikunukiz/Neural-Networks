import coco_proc, tester


if __name__ == '__main__':

    loc = 'models/mlbl_model.pkl'
    z, zd, zt = coco_proc.process(context=5)
    tester.tester(loc, z, zt)
