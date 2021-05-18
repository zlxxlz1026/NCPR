from RL.env_dialogue import DialogueEnv
from utils import *
from itertools import count
from RL_model import Agent,ReplayMemory
import argparse
import os

#os.environ['CUDA_VISIBLE_DEVICES']='2, 3, 4'
class Dialogue():

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

    def run(self):
        state = self.env.reset()
        state = torch.unsqueeze(torch.FloatTensor(state), 0).to(self.args.device)
        for t in count():
            action = self.agent.policy_net(state).max(1)[1].view(1, 1)
            next_state, reward, done = self.env.step(action.item())
            next_state = torch.tensor([next_state], device=self.args.device, dtype=torch.float)
            reward = torch.tensor([reward], device=self.args.device, dtype=torch.float)
            if done:
                next_state = None
                print("welcome next use,bye")
                break
            state = next_state

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

    model = Dialogue(args)
    model.run()