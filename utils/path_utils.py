class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if 'WHUHi' in dataset:
            strlist = str(dataset).split('_')
            trnval_path='/project/DW/Dataset/whuhi/Image/'+strlist[1]+'_'+strlist[2]+'/'

            return {'trnval':trnval_path}
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
