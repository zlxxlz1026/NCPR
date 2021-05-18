from queue import Queue
from threading import Thread
from RL_model import Agent,ReplayMemory
from utils import *
from RL.env_dialogue import DialogueEnv
import torch
import argparse
from zlx_socket import ServerSocket
from EmotionalAnalysis.emotional_analysis import EmotionalAnalysis
from RL.env_load_user import LoadUserInfoEnv
import json
import time


class producer(Thread):
    def __init__(self, server_socket, args):
        Thread.__init__(self)
        self.q_list = []
        self.server_socket = server_socket
        self.args = args

    def _append_queue(self, name, addr):
        tmp_q = Queue()
        self.q_list.append(tmp_q)
        tmp_t = consumer(tmp_q, name, addr, self.server_socket, args=self.args)
        tmp_t.start()

    def _welcome(self, addr, name):
        x = "welcome use zlxxlz's dialogue"
        msg = {
            "name":name,
            "msg": x,
        }
        print(json.dumps(msg))
        self.server_socket.udp_handle_send(addr, json.dumps(msg))
        # print("welcome use zlxxlz\'s dialogue")

    def run(self):
        while True:
            raw_json, addr = self.server_socket.listen()
            msg = {
                "raw_json":raw_json,
                "addr":addr,
            }
            if raw_json['type'] == 1:
                self._welcome(addr, raw_json['name'])
                self._append_queue(raw_json['name'], addr)
                print(f"new thread {raw_json['name']}")
            else:
                for q in self.q_list:
                    q.put(msg)


class consumer(Thread):
    def __init__(self, q, name, addr, server_socket, args):
        Thread.__init__(self)
        self.q = q
        self.name = name
        self.server_socket = server_socket
        self.reset_flag = True
        self.args = args
        self.addr = addr
        kg = load_kg(args.data_name)
        feature_length = len(kg.G['feature'].keys())
        args.attr_num = feature_length
        dataset = load_dataset(args.data_name)
        filename = 'train-data-{}-RL-command-{}-ask_method-{}-attr_num-{}-ob-{}'.format(
            args.data_name, args.command, args.entropy_method, args.attr_num, args.observe_num)
        if self.name.isdigit() and int(self.name) > 0 and int(self.name) < 1800:
            self.load_env = LoadUserInfoEnv(kg, dataset, args.data_name, seed=args.seed, max_turn=args.max_turn,
                                           cand_len_size=args.cand_len_size, attr_num=args.attr_num,
                                           command=args.command, ask_num=args.ask_num, entropy_way=args.entropy_method,
                                           fm_epoch=args.fm_epoch)
            state_space = self.load_env.state_space
            action_space = self.load_env.action_space
        else:
            self.env = DialogueEnv(kg, dataset, args.data_name, seed=args.seed, max_turn=args.max_turn,
                                           cand_len_size=args.cand_len_size, attr_num=args.attr_num,
                                           command=args.command, ask_num=args.ask_num, entropy_way=args.entropy_method,
                                           fm_epoch=args.fm_epoch)
            state_space = self.env.state_space
            action_space = self.env.action_space
        memory = ReplayMemory(args.memory_size)  # 10000
        self.agent = Agent(device=args.device, memory=memory, state_space=state_space, hidden_size=args.hidden, action_space=action_space)
        if args.load_rl_epoch != 0:
            print('Staring loading rl model in epoch {}'.format(args.load_rl_epoch))
            self.agent.load_model(data_name=args.data_name, filename=filename, epoch_user=args.load_rl_epoch)

        #一个标识句子，用来标识用户情感
        self.emotional_msg = ""

        #情感分析模型，用以分析用户的正负情感
        self.emotional_model = EmotionalAnalysis()

    def _ask_attribute(self, attr_list):
        x = self.emotional_msg + 'do you like' + str(attr_list)
        msg = {
            'name':self.name,
            'msg':x
        }
        self.server_socket.udp_handle_send(self.addr, json.dumps(msg))
        # print(f'Do you like {attr_list}?')

    def _recommend_items(self, item_list):
        x = self.emotional_msg + 'i guess you might like these:' + str(item_list)
        msg = {
            'name':self.name,
            'msg':x,
        }
        self.server_socket.udp_handle_send(self.addr, json.dumps(msg))
        # print(f'Here is your recommendation:{item_list}, do you like them ?')

    def _ask_again(self):
        x = "sorry, I don't know if you like it"
        msg = {
            'name':self.name,
            'msg':x,
        }
        self.server_socket.udp_handle_send(self.addr, json.dumps(msg))

    def _recommend_over(self):
        x = "Thank you, see you next time"
        msg = {
            'name':self.name,
            'msg':x,
        }
        self.server_socket.udp_handle_send(self.addr, json.dumps(msg))

    def _interaction(self):
        while True:
            msg = self.q.get()
            raw_json = msg['raw_json']
            addr = msg['addr']
            if raw_json['name'] != self.name:
                continue
            if raw_json['type'] == 0:
                if self.reset_flag:
                    self.state = self.env.reset(raw_json['message'])
                    self.state = torch.unsqueeze(torch.FloatTensor(self.state), 0).to(self.args.device)
                    self.action = self.agent.policy_net(self.state).max(1)[1].view(1, 1)
                    if self.action == 0:
                        attr_list = self.env.get_top_attr()
                        self._ask_attribute(attr_list)
                    elif self.action == 1:
                        items_list = self.env.get_top_items()
                        self._recommend_items(items_list)
                    self.reset_flag = False
                else:
                    emotional_flag = self.emotional_model.get_score(raw_json['message'])['items'][0]['sentiment']
                    if emotional_flag == 1:
                        self._ask_again()
                        continue
                    elif emotional_flag == 0:
                        tmp_flag = 'no'
                        self.emotional_msg = "I'm sorry to hear that, "
                    else:
                        tmp_flag = 'yes'
                        self.emotional_msg = "I'm glad you like it, "
                    self.state, reward, done = self.env.step(self.action.item(), tmp_flag)
                    self.state = torch.tensor([self.state], device=self.args.device, dtype=torch.float)
                    reward = torch.tensor([reward], device=self.args.device, dtype=torch.float)
                    if done:
                        self.state = None
                        print("Thank you, see you next time")
                        self._recommend_over()
                        return
                    self.action = self.agent.policy_net(self.state).max(1)[1].view(1, 1)
                    if self.action == 0:
                        attr_list = self.env.get_top_attr()
                        self._ask_attribute(attr_list)
                    elif self.action == 1:
                        items_list = self.env.get_top_items()
                        self._recommend_items(items_list)
            elif raw_json['type'] == 2:
                print(f'quit thread {self.name}')
                return

    def _print_user_info(self):
        for x in self.load_env.get_print_info():
            msg = {
                'name' : self.name,
                'msg' : x,
            }
            self.server_socket.udp_handle_send(self.addr, json.dumps(msg))
            time.sleep(0.5)

    def _load(self):
        self.state = self.load_env.reset(int(self.name))
        self.state = torch.unsqueeze(torch.FloatTensor(self.state), 0).to(args.device)
        # print(self.load_env.get_conversation_info())
        while True:
            self.action = self.agent.policy_net(self.state).max(1)[1].view(1, 1)
            self.state, reward, done = self.load_env.step(self.action.item())
            self.state = torch.tensor([self.state], device=self.args.device, dtype=torch.float)
            reward = torch.tensor([reward], device=self.args.device, dtype=torch.float)
            if done:
                break
            # tmp_s = self.load_env.get_conversation_his()
            # if tmp_s == 1:
            #     print(self.load_env.get_conversation_info())
            #     print("yes")
            # elif tmp_s == -1:
            #     print(self.load_env.get_conversation_info())
            #     print("no")
            # elif tmp_s == 2:
            #     print(self.load_env.get_conversation_info())
            #     print("yes")
            # elif tmp_s == -2:
            #     print(self.load_env.get_conversation_info())
            #     print("no")
        self._print_user_info()
        print(self.load_env.get_print_info())

    def run(self):
        if self.name.isdigit() and int(self.name) > 0 and int(self.name) < 1800:
            self._load()
        else:
            self._interaction()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-seed', type=int, default=1, help='random seed.')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--epochs', '-me', type=int, default=50000, help='the number of RL train epoch')
    parser.add_argument('--fm_epoch', type=int, default=245, help='the epoch of FM embedding')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size.')
    parser.add_argument('--gamma', type=float, default=0.999, help='reward discount factor.')
    parser.add_argument('--target_update', type=int, default=20, help='the number of epochs to update policy parameters')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate.')
    parser.add_argument('--hidden', type=int, default=512, help='number of samples')
    parser.add_argument('--memory_size', type=int, default=50000, help='size of memory ')
    parser.add_argument('--data_name', type=str, default=LAST_FM, choices=[LAST_FM, LAST_FM_STAR, YELP, YELP_STAR],
                        help='One of {LAST_FM, LAST_FM_STAR, YELP, YELP_STAR}.')
    parser.add_argument('--entropy_method', type=str, default='entropy', help='entropy_method is one of {entropy, weight entropy}')
    # Although the performance of 'weighted entropy' is better, 'entropy' is an alternative method considering the time cost.
    parser.add_argument('--max_turn', type=int, default=15, help='max conversation turn')
    parser.add_argument('--cand_len_size', type=int, default=20, help='binary state size for the length of candidate items')
    parser.add_argument('--attr_num', type=int, help='the number of attributes')
    parser.add_argument('--mode', type=str, default='train', help='the mode in [train, test]')
    parser.add_argument('--command', type=int, default=7, help='select state vector')
    parser.add_argument('--ask_num', type=int, default=1, help='the number of features asked in a turn')
    parser.add_argument('--observe_num', type=int, default=1000, help='the number of epochs to save RL model and metric')
    parser.add_argument('--load_rl_epoch', type=int, default=48000, help='the epoch of loading RL model')
    args = parser.parse_args()
    args.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    s = ServerSocket(8888,'127.0.0.1', 'udp')
    t = producer(s, args)
    t.start()
