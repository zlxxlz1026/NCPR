import os
from easydict import EasyDict as edict
import json

if __name__ == '__main__':
    dir = './data/lastfm_mine/FM-train-data-lastfm'
    files = edict(
        # train='review_dict_train.json',
        test='review_dict_test.json',
        valid='review_dict_valid.json',
    )
    mydict = {}
    for i in files:
        with open(os.path.join(dir,files[i]), encoding='utf-8') as f:
            tmp_dict = json.load(f)
            for k in tmp_dict.keys():
                if k not in mydict:
                    mydict[k] = tmp_dict[k]
                else:
                    mydict[k].extend(tmp_dict[k])
    for i in mydict.keys():
        mydict[i] = sorted(mydict[i])
    print(mydict)
    # val_dir = './data/lastfm/Graph_generate_data/user_item.json'
    # with open(val_dir, encoding='utf-8') as f:
    #     val_dict = json.load(f)
    # res = []
    # for i in mydict.keys():
    #     print(f'user: {i}, result: {mydict[i] == val_dict[i]}')
    #     res.append(mydict[i] == val_dict[i])
    # print(res.count(True))

    dir1 = './data/lastfm/UI_Interaction_data'
    files = edict(
        # train='review_dict_train.json',
        test='review_dict_test.json',
        valid='review_dict_valid.json',
    )
    mydict1 = {}
    for i in files:
        with open(os.path.join(dir1,files[i]), encoding='utf-8') as f:
            tmp_dict = json.load(f)
            for k in tmp_dict.keys():
                if k not in mydict1:
                    mydict1[k] = tmp_dict[k]
                else:
                    mydict1[k].extend(tmp_dict[k])
    for i in mydict1.keys():
        mydict1[i] = sorted(mydict1[i])
    print(mydict1)