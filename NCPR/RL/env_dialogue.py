
import json
import numpy as np
import os
import random
from utils import *
import itertools
from tkinter import _flatten
from collections import Counter
class DialogueEnv(object):
    def __init__(self, kg, dataset, data_name, seed=1, max_turn=15, cand_len_size=20, attr_num=20, command=1, ask_num=1, entropy_way='weight entropy', fm_epoch=250):
        self.data_name = data_name
        self.command = command
        self.seed = seed
        self.max_turn = max_turn    #MAX_TURN
        self.attr_state_num = attr_num
        self.cand_len_size = cand_len_size
        self.kg = kg
        self.dataset = dataset
        self.feature_length = getattr(self.dataset, 'feature').value_len
        self.user_length = getattr(self.dataset, 'user').value_len
        self.item_length = getattr(self.dataset, 'item').value_len

        # action parameters
        self.ask_num = ask_num
        self.rec_num = 10
        #  entropy  or weight entropy
        self.ent_way = entropy_way

        # user's profile
        self.reachable_feature = []   # user reachable feature
        self.user_acc_feature = []  # user accepted feature which asked by agent
        self.user_rej_feature = []  # user rejected feature which asked by agent
        self.cand_items = []   # candidate items

        #user_id  item_id   cur_step   cur_node_set
        self.user_id = None
        self.target_item = None
        self.cur_conver_step = 0        #  the number of conversation in current step
        self.cur_node_set = []     # maybe a node or a node set  /   normally save feature node
        # state veactor
        self.user_embed = None
        self.conver_his = []    #conversation_history
        self.cand_len = []    #the number of candidate items  [binary ]
        self.attr_ent = []  # attribute entropy

        # self.ui_dict = self.__load_rl_data__(data_name, mode=mode)  # np.array [ u i weight]
        self.user_weight_dict = dict()
        self.user_items_dict = dict()

        #init seed & init user_dict
        set_random_seed(self.seed) # set random seed
        print(f'fm_epoch : {fm_epoch}')
        embeds = load_embed(data_name, epoch=fm_epoch)
        self.ui_embeds =embeds['ui_emb']
        self.feature_emb = embeds['feature_emb']
        self.action_space = 2
        self.state_space_dict = {
            1: self.max_turn + self.cand_len_size + self.attr_state_num + self.ui_embeds.shape[1],
            2: self.attr_state_num,  # attr_ent
            3: self.max_turn,  #conver_his
            4: self.cand_len_size,  #cand_item
            5: self.ui_embeds.shape[1], # user_embedding
            6: self.cand_len_size + self.attr_state_num + self.max_turn, #attr_ent + conver_his + cand_item
            7: self.cand_len_size + self.max_turn,
        }
        self.state_space = self.state_space_dict[self.command]
        self.reward_dict = {
            'ask_suc': 0.01,
            'ask_fail': -0.1,
            'rec_suc': 1,
            'rec_fail': -0.1,
            'until_T': -0.3,      # MAX_Turn
            'cand_none': -0.1
        }
        self.history_dict = {
            'ask_suc': 1,
            'ask_fail': -1,
            'rec_scu': 2,
            'rec_fail': -2,
            'until_T': 0
        }
        self.attr_count_dict = dict()   # This dict is used to calculate entropy
        self.tag2name = dict()
        self.index2name = dict()
        #TODO:思考一下文件调用位置是相对于调用的代码还是当前代码
        self.__load_map_data('./data/lastfm/Map_data/')

    def __load_map_data(self,dir):
        with open(os.path.join(dir,'tag_2_name.json'),'r',encoding='utf-8') as f:
            self.tag2name = json.load(f)
        with open(os.path.join(dir,'index_2_name.json'),'r',encoding='utf-8') as f:
            self.index2name = json.load(f)


    def reset(self, attribute):
        #init  user_id  item_id  cur_step   cur_node_set
        self.cur_conver_step = 0   #reset cur_conversation step
        #节点set
        self.cur_node_set = []
        self.reachable_feature = []  # user reachable feature in cur_step
        self.user_acc_feature = []  # user accepted feature which asked by agent
        self.user_rej_feature = []  # user rejected feature which asked by agent
        self.cand_items = list(range(self.item_length))

        # init  state vector
        # self.user_embed = self.ui_embeds[self.user_id].tolist()  # init user_embed   np.array---list
        self.conver_his = [0] * self.max_turn  # conversation_history
        self.cand_len = [self.feature_length >> d & 1 for d in range(self.cand_len_size)][::-1]  #Binary representation of candidate set length
        self.attr_ent = [0] * self.attr_state_num  # attribute entropy

        self._updata_reachable_feature(start='user', attribute=attribute)
        self.reachable_feature = list(set(self.reachable_feature) - set(self.user_acc_feature))
        self.conver_his[self.cur_conver_step] = self.history_dict['ask_suc']
        self.cur_conver_step += 1

        print('=== init user prefer feature: {}'.format(self.cur_node_set))
        self._update_cand_items(acc_feature=self.cur_node_set, rej_feature=[])
        self._update_feature_entropy()  #update entropy
        print('reset_reachable_feature num: {}'.format(len(self.reachable_feature)))

        # Sort reachable features according to the entropy of features
        reach_fea_score = self._feature_score()
        max_ind_list = []
        for k in range(self.ask_num):
            max_score = max(reach_fea_score)
            max_ind = reach_fea_score.index(max_score)
            reach_fea_score[max_ind] = 0
            max_ind_list.append(max_ind)
        max_fea_id = [self.reachable_feature[i] for i in max_ind_list]
        [self.reachable_feature.pop(v - i) for i, v in enumerate(max_ind_list)]
        [self.reachable_feature.insert(0, v) for v in max_fea_id[::-1]]

        return self._get_state()

    def _get_state(self):
        if self.command == 1:
            state = [self.user_embed, self.conver_his, self.attr_ent, self.cand_len]
            state = list(_flatten(state))
        elif self.command == 2: #attr_ent
            state = self.attr_ent
            state = list(_flatten(state))
        elif self.command == 3: #conver_his
            state = self.conver_his
            state = list(_flatten(state))
        elif self.command == 4: #cand_len
            state = self.cand_len
            state = list(_flatten(state))
        elif self.command == 5:  #user_embedding
            state = self.user_embed
            state = list(_flatten(state))
        elif self.command == 6: #attr_ent + conver_his + cand_len
            state = [self.conver_his, self.attr_ent, self.cand_len]
            state = list(_flatten(state))
        elif self.command == 7: #conver_his + cand_len
            state = [self.conver_his, self.cand_len]
            state = list(_flatten(state))
        return state

    #flag表示接受的用户偏好(yes or no)
    def step(self, action, flag):   #action:0  ask   action:1  recommend   setp=MAX_TURN  done
        done = 0
        print('---------------step:{}-------------'.format(self.cur_conver_step))

        if self.cur_conver_step == self.max_turn:
            reward = self.reward_dict['until_T']
            self.conver_his[self.cur_conver_step-1] = self.history_dict['until_T']
            print('--> Maximum number of turns reached !')
            done = 1
        elif action == 0:   #ask feature
            print('-->action: ask features')
            reward, done, acc_feature, rej_feature = self._ask_update(flag)  #update user's profile:  user_acc_feature & user_rej_feature
            self._update_cand_items(acc_feature, rej_feature)   #update cand_items

            if len(acc_feature):   # can reach new feature：  update current node and reachable_feature
                self.cur_node_set = acc_feature
                self._updata_reachable_feature(start='feature')  # update user's profile: reachable_feature

            self.reachable_feature = list(set(self.reachable_feature) - set(self.user_acc_feature))
            self.reachable_feature = list(set(self.reachable_feature) - set(self.user_rej_feature))

            if self.command in [1, 2, 6, 7]:  # update attr_ent
                self._update_feature_entropy()
            if len(self.reachable_feature) != 0:  # if reachable_feature == 0 :cand_item= 1
                reach_fea_score = self._feature_score()  # compute feature score

                max_ind_list = []
                for k in range(self.ask_num):
                    max_score = max(reach_fea_score)
                    max_ind = reach_fea_score.index(max_score)
                    reach_fea_score[max_ind] = 0
                    max_ind_list.append(max_ind)
                max_fea_id = [self.reachable_feature[i] for i in max_ind_list]
                [self.reachable_feature.pop(v - i) for i, v in enumerate(max_ind_list)]
                [self.reachable_feature.insert(0, v) for v in max_fea_id[::-1]]

        elif action == 1:  #recommend items
            #select topk candidate items to recommend
            # cand_item_score = self._item_score()
            # item_score_tuple = list(zip(self.cand_items, cand_item_score))
            # sort_tuple = sorted(item_score_tuple, key=lambda x: x[1], reverse=True)
            # self.cand_items, cand_item_score = zip(*sort_tuple)


            #===================== rec update=========
            reward, done = self._recommend_updata(flag)
            #========================================
            if reward == 1:
                print('-->Recommend successfully!')
            else:
                if self.command in [1, 2, 6, 7]:  # update attr_ent
                    self._update_feature_entropy()
                print('-->Recommend fail !')


        self.cur_conver_step += 1
        return self._get_state(), reward, done

    def _updata_reachable_feature(self, start='feature', attribute = ''):
        #self.reachable_feature = []
        if start == 'user':
            user_like_fea = int(attribute)
            # user_like_random_fea = random.choice(self.kg.G['item'][self.target_item]['belong_to'])
            self.user_acc_feature.append(user_like_fea) #update user acc_fea
            self.cur_node_set = [user_like_fea]

            next_reachable_feature = []
            for cur_node in self.cur_node_set:
                fea_belong_items = list(self.kg.G['feature'][cur_node]['belong_to'])  # A-I
                fea_like_users = list(self.kg.G['feature'][cur_node]['like'])  # A-U

                if self.data_name == 'lastfm':
                    # update reachable feature
                    # user_friends = self.kg.G['user'][self.user_id]['friends']
                    cand_fea_like_users = list(set(fea_like_users))
                    for user_id in cand_fea_like_users:  # A-U-A  # U in [friends]
                        next_reachable_feature.append(list(self.kg.G['user'][user_id]['like']))
                    next_reachable_feature = list(set(_flatten(next_reachable_feature)))
                #含偏好属性的物品集合
                cand_fea_belong_items = list(set(fea_belong_items) & set(self.cand_items))

                for item_id in cand_fea_belong_items:  # A-I-A   I in [cand_items]
                    next_reachable_feature.append(list(self.kg.G['item'][item_id]['belong_to']))
                next_reachable_feature = list(set(_flatten(next_reachable_feature)))
            self.reachable_feature = next_reachable_feature  # init reachable_feature

        elif start == 'feature':
            next_reachable_feature = []
            for cur_node in self.cur_node_set:
                fea_belong_items = list(self.kg.G['feature'][cur_node]['belong_to']) # A-I
                fea_like_users = list(self.kg.G['feature'][cur_node]['like'])   #A-U

                if self.data_name == 'lastfm':
                # update reachable feature
                #     user_friends = self.kg.G['user'][self.user_id]['friends']
                    cand_fea_like_users = list(set(fea_like_users))
                    for user_id in cand_fea_like_users:  # A-U-A  # U in [friends]
                        next_reachable_feature.append(list(self.kg.G['user'][user_id]['like']))
                    next_reachable_feature = list(set(_flatten(next_reachable_feature)))

                cand_fea_belong_items = list(set(fea_belong_items) & set(self.cand_items))
                for item_id in cand_fea_belong_items:  # A-I-A   I in [cand_items]
                    next_reachable_feature.append(list(self.kg.G['item'][item_id]['belong_to']))
                next_reachable_feature = list(set(_flatten(next_reachable_feature)))
            self.reachable_feature = next_reachable_feature


    def _feature_score(self):
        reach_fea_score = []
        for feature_id in self.reachable_feature:
            score = self.attr_ent[feature_id]
            reach_fea_score.append(score)
        return reach_fea_score

    def _item_score(self):
        cand_item_score = []
        for item_id in self.cand_items:
            item_embed = self.ui_embeds[self.user_length + item_id]
            score = 0
            # score += np.inner(np.array(self.user_embed), item_embed)
            prefer_embed = self.feature_emb[self.user_acc_feature, :]  #np.array (x*64)
            for i in range(len(self.user_acc_feature)):
                score += np.inner(prefer_embed[i], item_embed)
            cand_item_score.append(score)
        return cand_item_score


    def _ask_update(self, flag):
        '''
        :return: reward, acc_feature, rej_feature
        '''
        done = 0
        # TODO datafram!     groundTruth == target_item features
        # feature_groundtrue = self.kg.G['item'][self.target_item]['belong_to']

        remove_acced_reachable_fea = self.reachable_feature.copy()  # copy reachable_feature
        ask_feature = remove_acced_reachable_fea[:self.ask_num]
        # print(f'Do you like {[self.tag2name[str(x)] for x in ask_feature]}?')
        # flag = input()
        if flag == "yes":
            acc_feature = list(ask_feature)
            rej_feature = []
        else:
            acc_feature = []
            rej_feature = list(ask_feature)
        #这里使用熵对所有属性排了序，因此如果询问就是index+1
        # acc_feature = list(set(remove_acced_reachable_fea[:self.ask_num]) & set(feature_groundtrue))
        # rej_feature = list(set(remove_acced_reachable_fea[:self.ask_num]) - set(acc_feature))

        #update user_acc_feature & user_rej_feature
        self.user_acc_feature.append(acc_feature)
        self.user_acc_feature = list(set(_flatten(self.user_acc_feature)))
        self.user_rej_feature.append(rej_feature)
        self.user_rej_feature = list(set(_flatten(self.user_rej_feature)))


        if len(acc_feature):
            reward = self.reward_dict['ask_suc']
            self.conver_his[self.cur_conver_step] = self.history_dict['ask_suc']   #update conver_his
        else:
            reward = self.reward_dict['ask_fail']
            self.conver_his[self.cur_conver_step] = self.history_dict['ask_fail']  #update conver_his

        if self.cand_items == []:  #candidate items is empty
            done = 1
            reward = self.reward_dict['cand_none']
        return reward, done, acc_feature, rej_feature

    def _update_cand_items(self, acc_feature, rej_feature):
        if len(acc_feature):    #accept feature
            print('=== ask acc: update cand_items')
            for feature_id in acc_feature:
                feature_items = self.kg.G['feature'][feature_id]['belong_to']
                self.cand_items = set(self.cand_items) & set(feature_items)   #  itersection
            self.cand_items = list(self.cand_items)

        self.cand_len = [len(self.cand_items) >>d & 1 for d in range(self.cand_len_size)][::-1]  # binary

    def _recommend_updata(self, rec_flag):
        print('-->action: recommend items')
        # recom_items = self.cand_items[: self.rec_num]    # TOP k item to recommend
        # print(f'Here is your recommendation:{[self.index2name[str(x)] for x in recom_items]}, do you like them ?')
        if rec_flag == "yes":
            reward = self.reward_dict['rec_suc']
            self.conver_his[self.cur_conver_step] = self.history_dict['rec_scu'] #update state vector: conver_his
            done = 1
        else:
            reward = self.reward_dict['rec_fail']
            self.conver_his[self.cur_conver_step] = self.history_dict['rec_fail']  #update state vector: conver_his
            if len(self.cand_items) > self.rec_num:
                self.cand_items = self.cand_items[self.rec_num:]  #update candidate items
            self.cand_len = [len(self.cand_items) >> d & 1 for d in range(self.cand_len_size)][::-1]  #  binary
            done = 0
        return reward, done

    def _update_feature_entropy(self):
        if self.ent_way == 'entropy':
            cand_items_fea_list = []
            for item_id in self.cand_items:
                cand_items_fea_list.append(list(self.kg.G['item'][item_id]['belong_to']))
            cand_items_fea_list = list(_flatten(cand_items_fea_list))
            self.attr_count_dict = dict(Counter(cand_items_fea_list))
            self.attr_ent = [0] * self.attr_state_num  # reset attr_ent
            real_ask_able = list(set(self.reachable_feature) & set(self.attr_count_dict.keys()))
            for fea_id in real_ask_able:
                #这里除以候选物品集长度是因为每个特征在每个物品中最多出现一次
                p1 = float(self.attr_count_dict[fea_id]) / len(self.cand_items)
                p2 = 1.0 - p1
                if p1 == 1:
                    self.attr_ent[fea_id] = 0
                else:
                    ent = (- p1 * np.log2(p1) - p2 * np.log2(p2))
                    self.attr_ent[fea_id] = ent

        elif self.ent_way == 'weight entropy':
            cand_items_fea_list = []
            self.attr_count_dict = {}
            cand_item_score = self._item_score()
            cand_item_score_sig = self.sigmoid(cand_item_score)  # sigmoid(score)
            for score_ind, item_id in enumerate(self.cand_items):
                cand_items_fea_list = list(self.kg.G['item'][item_id]['belong_to'])
                for fea_id in cand_items_fea_list:
                    if self.attr_count_dict.get(fea_id) == None:
                        self.attr_count_dict[fea_id] = 0
                    self.attr_count_dict[fea_id] += cand_item_score_sig[score_ind]

            self.attr_ent = [0] * self.attr_state_num  # reset attr_ent
            real_ask_able = list(set(self.reachable_feature) & set(self.attr_count_dict.keys()))
            sum_score_sig = sum(cand_item_score_sig)

            for fea_id in real_ask_able:
                p1 = float(self.attr_count_dict[fea_id]) / sum_score_sig
                p2 = 1.0 - p1
                if p1 == 1 or p1 <= 0:
                    self.attr_ent[fea_id] = 0
                else:
                    ent = (- p1 * np.log2(p1) - p2 * np.log2(p2))
                    self.attr_ent[fea_id] = ent

    def sigmoid(self, x_list):
        x_np = np.array(x_list)
        s = 1 / (1 + np.exp(-x_np))
        return s.tolist()

    def get_top_attr(self):
        remove_acced_reachable_fea = self.reachable_feature.copy()  # copy reachable_feature
        ask_feature = remove_acced_reachable_fea[:self.ask_num]
        return [self.tag2name[str(x)] for x in ask_feature]

    def get_top_items(self):
        cand_item_score = self._item_score()
        item_score_tuple = list(zip(self.cand_items, cand_item_score))
        sort_tuple = sorted(item_score_tuple, key=lambda x: x[1], reverse=True)
        self.cand_items, cand_item_score = zip(*sort_tuple)
        recom_items = self.cand_items[: self.rec_num]
        return [self.index2name[str(x)] for x in recom_items]
