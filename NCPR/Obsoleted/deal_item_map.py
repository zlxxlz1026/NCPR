import os
import json
import pandas as pd

if __name__ == '__main__':
    dir = './data/lastfm/Map_data/'
    map2index = {}
    with open(os.path.join(dir,'item_map.json')) as f:
        map2index = json.load(f)
    map2name = pd.read_csv(os.path.join(dir,'artists.dat'),sep='\t').iloc[:,:2]
    map2name = map2name.set_index('id').T.to_dict()
    print(len(map2name.keys()))
    index_list = list(map2index.keys())
    print(index_list)
    index2name = {}
    for i in index_list:
        k = int(i)
        if k in map2name.keys():
            index2name[map2index[i]] = map2name[k]['name']
        else:
            # print(k)
            index2name[map2index[i]] = k
    print(index2name)
    with open(os.path.join(dir,'index_2_name.json'), 'w+', encoding='utf-8') as f:
        json.dump(index2name,f,indent=4, ensure_ascii = False)
