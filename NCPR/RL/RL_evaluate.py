import time
import argparse
from itertools import count
import torch.nn as nn
import torch
from collections import namedtuple
from utils import *
from RL.env_binary_question import BinaryRecommendEnv
# from RL.env_lfpan import LFPANEnv
# from RL.env_yfpan import RFAPNEnv
from RL.env_enumerated_question import EnumeratedRecommendEnv
import copy

EnvDict = {
        LAST_FM: BinaryRecommendEnv,
        # LAST_FM: LFPANEnv,
        LAST_FM_STAR: BinaryRecommendEnv,
        YELP: EnumeratedRecommendEnv,
        # YELP: RFAPNEnv,
        YELP_STAR: BinaryRecommendEnv
    }

def dqn_evaluate(args, kg, dataset, agent, filename, i_episode):
    test_env = EnvDict[args.data_name](kg, dataset, args.data_name, seed=args.seed, max_turn=args.max_turn,
                                       cand_len_size=args.cand_len_size, attr_num=args.attr_num, mode='test',
                                       command=args.command, ask_num=args.ask_num, entropy_way=args.entropy_method,
                                       fm_epoch=args.fm_epoch, hyper=args.hyper, method=args.method, update=True)
    agent.load_model(data_name=args.data_name, filename=filename, epoch_user=i_episode, hyper=args.hyper, method=args.method)
    set_random_seed(args.seed)
    # tmp_agent = copy.deepcopy(agent)
    tt = time.time()
    start = tt
    # self.reward_dict = {
    #     'ask_suc': 0.1,
    #     'ask_fail': -0.1,
    #     'rec_suc': 1,
    #     'rec_fail': -0.3,
    #     'until_T': -0.3,  # until MAX_Turn
    #     'cand_none': -0.1
    # }
    # ealuation metric  ST@T
    SR5, SR10, SR15, AvgT = 0, 0, 0, 0
    SR_turn_15 = [0]* args.max_turn
    turn_result = []
    result = []
    user_size = test_env.ui_array.shape[0]
    print('User size in UI_test: ', user_size)
    test_filename = 'Evaluate-epoch-{}-'.format(i_episode) + filename
    if args.data_name in [LAST_FM_STAR, LAST_FM]:
        test_size = 4000   # Only do 4000 iteration for the sake of time
        user_size = test_size
    if args.data_name in [YELP_STAR, YELP]:
        test_size = 2500     # Only do 2500 iteration for the sake of time
        user_size = test_size
    print('The select Test size : ', test_size)
    for user_num in range(user_size+1):  #user_size
        # TODO uncommend this line to print the dialog process
        blockPrint()
        print('\n================test tuple:{}===================='.format(user_num))
        state = test_env.reset()  # Reset environment and record the starting state
        state = torch.unsqueeze(torch.FloatTensor(state), 0).to(args.device)
        for t in count():  # user  dialog
            action = agent.policy_net(state).max(1)[1].view(1, 1)
            next_state, reward, done = test_env.step(action.item())
            next_state = torch.tensor([next_state], device=args.device, dtype=torch.float)
            reward = torch.tensor([reward], device=args.device, dtype=torch.float)
            if done:
                next_state = None
            state = next_state
            if done:
                enablePrint()
                if reward.item() == 1:  # recommend successfully
                    SR_turn_15 = [v+1 if i>t  else v for i, v in enumerate(SR_turn_15) ]
                    if t < 5:
                        SR5 += 1
                        SR10 += 1
                        SR15 += 1
                    elif t < 10:
                        SR10 += 1
                        SR15 += 1
                    else:
                        SR15 += 1

                AvgT += t+1
                break
        enablePrint()
        if user_num % args.observe_num == 0 and user_num > 0:
            SR = [SR5/args.observe_num, SR10/args.observe_num, SR15/args.observe_num, AvgT / args.observe_num]
            SR_TURN = [i/args.observe_num for i in SR_turn_15]
            print('Total evalueation epoch_uesr:{}'.format(user_num + 1))
            print('Takes {} seconds to finish {}% of this task'.format(str(time.time() - start),
                                                                       float(user_num) * 100 / user_size))
            print('SR5:{}, SR10:{}, SR15:{}, AvgT:{} '
                  'Total epoch_uesr:{}'.format(SR5 / args.observe_num, SR10 / args.observe_num, SR15 / args.observe_num,
                                                AvgT / args.observe_num, user_num + 1))
            result.append(SR)
            turn_result.append(SR_TURN)
            SR5, SR10, SR15, AvgT = 0, 0, 0, 0
            SR_turn_15 = [0] * args.max_turn
            tt = time.time()

    SR5_mean = np.mean(np.array([item[0] for item in result]))
    SR10_mean = np.mean(np.array([item[1] for item in result]))
    SR15_mean = np.mean(np.array([item[2] for item in result]))
    AvgT_mean = np.mean(np.array([item[3] for item in result]))
    SR_all = [SR5_mean, SR10_mean, SR15_mean, AvgT_mean]
    save_rl_mtric(dataset=args.data_name, filename=filename, epoch=user_num, SR=SR_all, spend_time=time.time() - start,
                  mode='test', hyper=args.hyper, method=args.method)
    save_rl_mtric(dataset=args.data_name, filename=test_filename, epoch=user_num, SR=SR_all, spend_time=time.time() - start,
                  mode='test', hyper=args.hyper, method=args.method)  # save RL SR
    print('save test evaluate successfully!')

    SRturn_all = [0] * args.max_turn
    for i in range(len(SRturn_all)):
        SRturn_all[i] = np.mean(np.array([item[i] for item in turn_result]))
    print('success turn:{}'.format(SRturn_all))
    print(test_env.get_timer())
    t = args.method + '-' + str(args.hyper)
    PATH = f"{TMP_DIR[args.data_name]}/RL-log-merge/{t}/{test_filename}.txt"
    # PATH = TMP_DIR[args.data_name] + '/RL-log-merge/' + test_filename + '.txt'
    with open(PATH, 'a') as f:
        f.write('Training epocch:{}\n'.format(i_episode))
        f.write('===========Test Turn===============\n')
        f.write('Testing {} user tuples\n'.format(user_num))
        for i in range(len(SRturn_all)):
            f.write('Testing SR-turn@{}: {}\n'.format(i, SRturn_all[i]))
        f.write('================================\n')

