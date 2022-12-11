import copy
import time
# 参考：https://louris.cn/2022/04/19/dive-into-reinforcement-learning.html/
class CliffWalkingEnv:
    """ Cliff Walking环境"""

    def __init__(self, ncol=12, nrow=4):
        self.ncol = ncol  # 定义环境的宽
        self.nrow = nrow  # 定义环境的高
        self.P = self.createP()  # 转移矩阵P[state][action] = [(p, next_state, reward, done)]，包含下一个状态和奖励

    def createP(self):
        P = [[[] for j in range(4)] for i in range(self.nrow * self.ncol)]  # 初始化
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]  # 4 种动作, 0:上, 1:下, 2:左, 3:右。原点(0,0)定义在左上角
        for i in range(self.nrow):
            for j in range(self.ncol):
                for a in range(4):
                    if i == self.nrow - 1 and j > 0:  # 位置在悬崖或者终点，因为无法继续交互，任何动作奖励都为0
                        P[i * self.ncol + j][a] = [(1, i * self.ncol + j, 0, True)]
                        continue
                    # 其他位置
                    next_x = min(self.ncol - 1, max(0, j + change[a][0]))
                    next_y = min(self.nrow - 1, max(0, i + change[a][1]))
                    next_state = next_y * self.ncol + next_x
                    reward = -1
                    done = False
                    if next_y == self.nrow - 1 and next_x > 0:  # 下一个位置在悬崖或者终点
                        done = True
                        if next_x != self.ncol - 1:  # 下一个位置在悬崖
                            reward = -100
                    P[i * self.ncol + j][a] = [(1, next_state, reward, done)]
        return P


# 策略迭代
class PolicyIteration:
    """策略迭代"""

    def __init__(self, env, theta, gamma):
        self.env = env
        self.v = [0] * self.env.ncol * self.env.nrow  # 初始化价值为0
        self.pi = [[0.25, 0.25, 0.25, 0.25] for i in range(self.env.ncol * self.env.nrow)]  # 初始化为均匀随机策略
        self.theta = theta  # 策略评估收敛阈值
        self.gamma = gamma  # 折扣因子

    def policy_evaluation(self):  # 策略评估
        cnt = 1  # 计数器
        while True:
            max_diff = 0
            new_v = [0] * self.env.ncol * self.env.nrow
            for s in range(self.env.ncol * self.env.nrow):
                qsa_list = []
                for a in range(4):
                    qsa = 0
                    for res in self.env.P[s][a]:
                        p, next_state, r, done = res
                        # 本章节环境比较特殊，奖励和下一个状态有关，所以需要和状态转移概率相乘
                        qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                    qsa_list.append(self.pi[s][a] * qsa)
                new_v[s] = sum(qsa_list)  # 状态价值函数和动作价值函数之间的关系
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
            self.v = new_v
            if max_diff < self.theta:
                break
            cnt += 1
        print("策略评估进行%d轮后完成" % cnt)

    def policy_improvement(self):  # 策略提升
        for s in range(self.env.nrow * self.env.ncol):
            qsa_list = []
            for a in range(4):
                qsa = 0
                for res in self.env.P[s][a]:
                    p, next_state, r, done = res
                    qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                qsa_list.append(qsa)
            maxq = max(qsa_list)
            cntq = qsa_list.count(maxq)
            self.pi[s] = [1 / cntq if q == maxq else 0 for q in qsa_list]  # 让相同的动作价值均分概率
        print("策略提升完成")
        return self.pi

    def policy_iteration(self):  # 策略迭代
        while True:
            self.policy_evaluation()
            old_pi = copy.deepcopy(self.pi)  # 将列表进行深拷贝，方便接下来进行比较
            new_pi = self.policy_improvement()
            print("policy_iteration", old_pi)
            if old_pi == new_pi:
                break


def print_agent(agent, action_meaning, disaster=[], end=[]):
    print("状态价值：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            print("%6.6s" % ("%.3f" % agent.v[i * agent.env.ncol + j]), end=" ")   # 为了输出美观，保持输出6个字符
        print()

    print("策略：")
    for i in range(agent.env.nrow):
        for j in range(agent.env.ncol):
            if (i * agent.env.ncol + j) in disaster:  # 一些特殊的状态，例如Cliff Walking中的悬崖
                print("****", end=" ")
            elif(i * agent.env.ncol + j) in end:  # 终点
                print("EEEE", end=" ")
            else:
                a = agent.pi[i * agent.env.ncol + j]
                pi_str = ""
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=" ")
        print()


# env = CliffWalkingEnv()
# action_meaning = ['^', 'v', '<', '>']
# theta = 0.001
# gamma = 0.9
# agent = PolicyIteration(env, theta, gamma)
# agent.policy_iteration()
# print_agent(agent, action_meaning, list(range(37, 47)), [47])
# print('-------------------CliffWalkingEnv-------------------')

# 价值迭代
class ValueIteration:
    """价值迭代"""
    def __init__(self, env, theta, gamma):
        self.env = env
        self.v = [0] * self.env.ncol * self.env.nrow  # 初始化价值为0
        self.theta = theta  # 价值收敛阈值
        self.gamma = gamma
        self.pi = [None for i in range(self.env.ncol * self.env.nrow)]  # 价值迭代结束后得到的策略

    def value_iteration(self):
        cnt = 0
        while True:
            max_diff = 0
            new_v = [0] * self.env.ncol * self.env.nrow
            for s in range(self.env.ncol * self.env.nrow):
                qsa_list = []
                for a in range(4):
                    qsa = 0
                    for res in self.env.P[s][a]:
                        p, next_state, r, done = res
                        qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                    qsa_list.append(qsa)   # 这一行和下一行是和策略迭代的主要区别
                new_v[s] = max(qsa_list)
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
            self.v = new_v
            if max_diff < self.theta:
                break
            cnt += 1
        print("价值迭代一共进行%d轮" % cnt)
        self.get_policy()

    def get_policy(self):   # 根据价值函数导出一个贪心策略
        for s in range(self.env.nrow * self.env.ncol):
            qsa_list = []
            for a in range(4):
                qsa = 0
                for res in self.env.P[s][a]:
                    p, next_state, r, done = res
                    qsa += r + p * self.gamma * self.v[next_state] * (1 - done)
                qsa_list.append(qsa)
            maxq = max(qsa_list)
            cntq = qsa_list.count(maxq)
            self.pi[s] = [1 / cntq if q == maxq else 0 for q in qsa_list]   # 让相同的动作价值均分概率


# env = CliffWalkingEnv()
# action_meaning = ['^', 'v', '<', '>']
# theta = 0.001
# gamma = 0.9
# agent = ValueIteration(env, theta, gamma)
# agent.value_iteration()
# print_agent(agent, action_meaning, list(range(37, 47)), [47])


import gym

env = gym.make("FrozenLake-v1", render_mode='human')  # 创建环境, gym==0.19.0版本已经更新了FrozenLake的版本
env = env.unwrapped  # 解封装才能访问状态转移矩阵P
env.reset()
env.render()    # 环境渲染，通常是弹窗显示或打印出可视化的环境

holes = set()
ends = set()

for s in env.P:
    for a in env.P[s]:
        for s_ in env.P[s][a]:
            if s_[2] == 1.0:  # 获得奖励为1，代表是终点
                ends.add(s_[1])
            if s_[3] is True:
                holes.add(s_[1])
holes = holes - ends
print("冰洞的索引：", holes)
print("终点的索引", ends)

for a in env.P[14]:  # 查看终点左边一格的状态转移信息
    print(env.P[14][a])


action_meaning = ['<', 'v', '>', '^']  # 这个动作意义是gym库中对FrozenLake这个环境事先规定好的
theta = 1e-5
gamma = 0.9
agent = PolicyIteration(env, theta, gamma)
agent.policy_iteration()
print_agent(agent, action_meaning, [5, 7, 11, 12], [15])
print("-------------------------------")
agent = ValueIteration(env, theta, gamma)
agent.value_iteration()
print_agent(agent, action_meaning, [5, 7, 11, 12], [15])
time.sleep(600)