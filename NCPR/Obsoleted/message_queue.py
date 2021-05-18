from queue import Queue
from threading import Thread
from RL_model import Agent,ReplayMemory
from utils import *
from RL.env_dialogue import DialogueEnv
import torch
import argparse
import json
from zlx_socket import ServerSocket
from EmotionalAnalysis.emotional_analysis import EmotionalAnalysis


class producer(Thread):
    def __init__(self, server_socket, args):
        Thread.__init__(self)
        self.q_list = []
        self.server_socket = server_socket
        self.args = args

    def _append_queue(self, name):
        tmp_q = Queue()
        self.q_list.append(tmp_q)
        tmp_t = consumer(tmp_q, name, self.server_socket, args=self.args)
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
                self._append_queue(raw_json['name'])
                print(f"new thread {raw_json['name']}")
            else:
                for q in self.q_list:
                    q.put(msg)


class consumer(Thread):
    def __init__(self, q, name, server_socket, args):
        Thread.__init__(self)
        self.q = q
        self.name = name
        self.server_socket = server_socket
        self.reset_flag = True
        self.args = args
        kg = load_kg(args.data_name)
        feature_length = len(kg.G['feature'].keys())
        args.attr_num = feature_length
        dataset = load_dataset(args.data_name)
        filename = 'train-data-{}-RL-command-{}-ask_method-{}-attr_num-{}-ob-{}'.format(
            args.data_name, args.command, args.entropy_method, args.attr_num, args.observe_num)
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

        self.emotional_model = EmotionalAnalysis()

    def _ask_attribute(self, addr, attr_list):
        x = 'Do you like' + str(attr_list)
        msg = {
            'name':self.name,
            'msg':x
        }
        self.server_socket.udp_handle_send(addr, json.dumps(msg))
        # print(f'Do you like {attr_list}?')

    def _recommend_items(self, addr, item_list):
        x = 'Here is your recommendation:' + str(item_list)
        msg = {
            'name':self.name,
            'msg':x,
        }
        self.server_socket.udp_handle_send(addr, json.dumps(msg))
        # print(f'Here is your recommendation:{item_list}, do you like them ?')

    def _ask_again(self, addr):
        x = "sorry, I don't know if you like it"
        self.server_socket.udp_handle_send(addr, x)

    def run(self):
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
                        self._ask_attribute(addr, attr_list)
                    elif self.action == 1:
                        items_list = self.env.get_top_items()
                        self._recommend_items(addr, items_list)
                    self.reset_flag = False
                else:
                    emotional_flag = self.emotional_model.get_score(raw_json['message'])['items'][0]['sentiment']
                    if emotional_flag == 1:
                        self._ask_again(addr)
                        continue
                    elif emotional_flag == 0:
                        tmp_flag = 'no'
                    else:
                        tmp_flag = 'yes'
                    self.state, reward, done = self.env.step(self.action.item(), tmp_flag)
                    self.state = torch.tensor([self.state], device=self.args.device, dtype=torch.float)
                    reward = torch.tensor([reward], device=self.args.device, dtype=torch.float)
                    if done:
                        self.state = None
                        print("welcome next use,bye")
                        return
                    self.action = self.agent.policy_net(self.state).max(1)[1].view(1, 1)
                    if self.action == 0:
                        attr_list = self.env.get_top_attr()
                        self._ask_attribute(addr, attr_list)
                    elif self.action == 1:
                        items_list = self.env.get_top_items()
                        self._recommend_items(addr, items_list)
            elif raw_json['type'] == 2:
                print(f'quit thread {self.name}')
                return

            # raw_data = self.q.get()
            # if raw_data['name'] == self.name:
            #     self.server_socket.udp_handle_send(('127.0.0.1', 10001), f"{raw_data['name']} receive data")
            #     print(f"{raw_data['name']} receive {raw_data['message']}")


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
