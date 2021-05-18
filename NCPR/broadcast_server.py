from RL.env_dialogue import DialogueEnv
from utils import *
from itertools import count
from RL_model import Agent,ReplayMemory
import argparse
from zlx_socket import ServerSocket

class Server():
    #初始化模型和环境
    def __init__(self, args):
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

        self.udp_server_socket = ServerSocket(8888,'127.0.0.1', 'udp')
        #该标记表示新用户进入需要reset()
        self.reset_flag = False


    def _welcome(self, addr):
        msg = 'welcome use zlxxlz\'s dialogue'
        self.udp_server_socket.udp_handle_send(addr, msg)
        # print("welcome use zlxxlz\'s dialogue")

    def _ask_attribute(self, addr, attr_list):
        self.udp_server_socket.udp_handle_send(addr, 'Do you like' + str(attr_list))
        # print(f'Do you like {attr_list}?')

    def _recommend_items(self, addr, item_list):
        self.udp_server_socket.udp_handle_send(addr, 'Here is your recommendation:' + str(item_list))
        # print(f'Here is your recommendation:{item_list}, do you like them ?')

    def run(self):
        while True:
            raw_json, addr = self.udp_server_socket.listen()
            #代表新用户进入
            if raw_json['type'] == 1:
                self._welcome(addr)
                self.reset_flag = True
            elif raw_json['type'] == 0:
                if self.reset_flag == True:
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
                #该逻辑表示用户输入yes或no
                else:
                    self.state, reward, done = self.env.step(self.action.item(), raw_json['message'])
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

            else:
                print('unknown type...')
                continue



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

    model = Server(args)
    model.run()
