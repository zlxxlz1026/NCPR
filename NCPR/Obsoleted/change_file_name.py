import os


if __name__ == '__main__':
    dir = '../../../data/lixi/data/lastfm/FM_sample_data'
    file_list = os.listdir(dir)

    for i in file_list:
        tmp_list = i.split('-')
        cnt = tmp_list[-1].split('.')[0]
        # print(cnt)
        new_name = ''
        if(tmp_list[-2]=='train'):
            new_name = f'sample_fm_data_train-{cnt}.pkl'
        elif(tmp_list[-2]=='valid'):
            new_name = f'sample_fm_data_valid-{cnt}.pkl'
        print(new_name)
        os.rename(os.path.join(dir,i),os.path.join(dir,new_name))
    print(os.listdir(dir))
    # for i in file_list:
    #     new_name = i+'.pkl'
    #     os.rename(os.path.join(dir,i),os.path.join(dir,new_name))
