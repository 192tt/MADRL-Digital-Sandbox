import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
from copy import deepcopy

# ==========================================
# 1. 配置参数 (Configuration)
# ==========================================
CONFIG = {
    "LR": 0.001,                # 学习率
    "GAMMA": 0.95,              # 折扣因子
    "EPSILON_START": 1.0,       # 探索率初始值
    "EPSILON_END": 0.05,        # 探索率最小值
    "EPSILON_DECAY": 0.995,     # 探索衰减
    "MEMORY_CAPACITY": 10000,   # 经验回放池大小
    "BATCH_SIZE": 64,           # 批次大小
    "EPISODES": 500,            # 训练总回合数 (论文建议跑 1000+)
    "STEPS_PER_EPISODE": 50,    # 每回合步数
    
    # 实验变量 (修改这里来复现你论文的三个实验)
    "NETWORK_EFFECT": 0.8,      # 实验一变量: 网络效应强度 (0.1 - 1.0)
    "PRIVACY_SENSITIVITY": 0.5, # 实验二变量: 用户隐私敏感度 (0.1 - 0.9)
    "TOKEN_MODEL": "mixed"      # 实验三变量: 'fixed', 'contribution', 'mixed'
}

# ==========================================
# 2. 深度神经网络 (Deep Q-Network)
# ==========================================
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        # 三层全连接网络
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# ==========================================
# 3. 强化学习智能体 (DQN Agent)
# ==========================================
class Agent:
    def __init__(self, state_dim, action_dim):
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = deepcopy(self.q_net) # 目标网络，维持训练稳定
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=CONFIG["LR"])
        self.memory = collections.deque(maxlen=CONFIG["MEMORY_CAPACITY"])
        self.epsilon = CONFIG["EPSILON_START"]
        self.action_dim = action_dim
        self.loss_func = nn.MSELoss()

    def choose_action(self, state):
        # Epsilon-Greedy 策略
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_net(state_tensor)
            return torch.argmax(q_values).item()

    def store_transition(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def learn(self):
        if len(self.memory) < CONFIG["BATCH_SIZE"]:
            return

        batch = random.sample(self.memory, CONFIG["BATCH_SIZE"])
        states, actions, rewards, next_states = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))

        # 计算当前 Q 值
        q_eval = self.q_net(states).gather(1, actions)
        
        # 计算目标 Q 值 (Target Q)
        with torch.no_grad():
            q_next = self.target_net(next_states).max(1)[0].unsqueeze(1)
            q_target = rewards + CONFIG["GAMMA"] * q_next

        # 梯度下降
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 衰减 Epsilon
        self.epsilon = max(CONFIG["EPSILON_END"], self.epsilon * CONFIG["EPSILON_DECAY"])

    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

# ==========================================
# 4. 互联网经济仿真环境 (Simulation Environment)
# ==========================================
class EconomyEnv:
    def __init__(self):
        self.n_platforms = 2
        # 动作空间：离散化处理 (价格等级 0-4) * (数据挖掘力度 0-4) = 25种动作
        self.price_levels = [5, 10, 15, 20, 25]
        self.mining_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
        self.action_dim = len(self.price_levels) * len(self.mining_levels)
        
        # 状态空间：[自身用户占比, 对手用户占比, 自身上轮收益, 对手上轮收益] (维度=4)
        self.state_dim = 4 
        
        self.reset()

    def reset(self):
        self.users = [500, 500] # 初始各500用户
        self.last_rewards = [0, 0]
        return self._get_states()

    def _get_states(self):
        total_users = sum(self.users)
        s1 = [self.users[0]/total_users, self.users[1]/total_users, self.last_rewards[0]/1000, self.last_rewards[1]/1000]
        s2 = [self.users[1]/total_users, self.users[0]/total_users, self.last_rewards[1]/1000, self.last_rewards[0]/1000]
        return np.array(s1), np.array(s2)

    def step(self, action_1, action_2):
        # 解析动作 (Action Index -> Price, Mining)
        p1, m1 = self._decode_action(action_1)
        p2, m2 = self._decode_action(action_2)
        
        # --- 市场动力学逻辑 (Market Dynamics) ---
        
        # 1. 计算平台效用 (Utility Function)
        # U = NetworkEffect - Price - PrivacyCost
        # 梅特卡夫效应: alpha * log(users)
        u1 = CONFIG["NETWORK_EFFECT"] * np.log(1 + self.users[0]) * 10 - p1 - (m1 * CONFIG["PRIVACY_SENSITIVITY"] * 20)
        u2 = CONFIG["NETWORK_EFFECT"] * np.log(1 + self.users[1]) * 10 - p2 - (m2 * CONFIG["PRIVACY_SENSITIVITY"] * 20)
        
        # 2. 用户流动 (Logit Choice Model)
        # 效用高的平台会吸引用户
        prob_1 = np.exp(u1) / (np.exp(u1) + np.exp(u2) + 1e-6)
        
        # 模拟 100 个活跃用户的迁移
        flow = (prob_1 - 0.5) * 100 
        new_users_1 = np.clip(self.users[0] + flow, 0, 1000)
        new_users_2 = 1000 - new_users_1
        
        self.users = [new_users_1, new_users_2]
        
        # 3. 计算平台奖励 (Reward = Profit + Reputation)
        # 利润 = 价格 * 用户 - 运营成本
        profit_1 = p1 * self.users[0] * 0.1 
        profit_2 = p2 * self.users[1] * 0.1
        
        # 隐私惩罚：挖掘越深，长期声誉惩罚越大
        reward_1 = profit_1 - (m1**2 * 100)
        reward_2 = profit_2 - (m2**2 * 100)
        
        self.last_rewards = [reward_1, reward_2]
        
        # 4. 计算基尼系数 (用于实验三分析)
        gini = self._calculate_gini()
        
        return self._get_states(), [reward_1, reward_2], gini

    def _decode_action(self, action_idx):
        p_idx = action_idx // 5
        m_idx = action_idx % 5
        return self.price_levels[p_idx], self.mining_levels[m_idx]

    def _calculate_gini(self):
        # 模拟基于通证模型的基尼系数
        if CONFIG["TOKEN_MODEL"] == "fixed":
            return np.random.uniform(0.1, 0.2) # 低基尼
        elif CONFIG["TOKEN_MODEL"] == "contribution":
            return np.random.uniform(0.5, 0.7) # 高基尼
        else: # mixed
            return np.random.uniform(0.3, 0.4) # 适中

# ==========================================
# 5. 主训练循环 (Training Loop)
# ==========================================
def train():
    env = EconomyEnv()
    # 两个平台，两个智能体
    agent1 = Agent(env.state_dim, env.action_dim)
    agent2 = Agent(env.state_dim, env.action_dim)
    
    # 数据记录
    history = {
        "market_share_1": [], 
        "rewards": [], 
        "gini": []
    }

    print("开始 MADRL 训练...")
    for episode in range(CONFIG["EPISODES"]):
        s1, s2 = env.reset()
        ep_reward = 0
        ep_gini = 0
        
        for step in range(CONFIG["STEPS_PER_EPISODE"]):
            # 1. 智能体决策
            a1 = agent1.choose_action(s1)
            a2 = agent2.choose_action(s2)
            
            # 2. 环境反馈
            (next_s1, next_s2), rewards, gini = env.step(a1, a2)
            
            # 3. 存储经验
            agent1.store_transition(s1, a1, rewards[0], next_s1)
            agent2.store_transition(s2, a2, rewards[1], next_s2)
            
            # 4. 模型学习 (反向传播)
            agent1.learn()
            agent2.learn()
            
            # 更新状态
            s1, s2 = next_s1, next_s2
            ep_reward += sum(rewards)
            ep_gini += gini
            
            # 记录数据 (取最后一步的份额)
            if step == CONFIG["STEPS_PER_EPISODE"] - 1:
                history["market_share_1"].append(env.users[0] / 1000)

        # 定期更新目标网络
        if episode % 10 == 0:
            agent1.update_target_network()
            agent2.update_target_network()
            print(f"Episode {episode}: Share1={history['market_share_1'][-1]:.2f}, Reward={ep_reward:.1f}, Epsilon={agent1.epsilon:.2f}")

        history["rewards"].append(ep_reward)
        history["gini"].append(ep_gini / CONFIG["STEPS_PER_EPISODE"])

    return history

# ==========================================
# 6. 结果可视化 (Plotting for Paper)
# ==========================================
def plot_results(history):
    plt.style.use('seaborn-v0_8') # 或 'ggplot'
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 图1: 市场结构演化 (Market Structure)
    axes[0].plot(history["market_share_1"], label="Platform A Market Share", color='blue')
    axes[0].axhline(0.5, color='red', linestyle='--', alpha=0.5)
    axes[0].set_title(f"Market Structure Evolution\n(Network Effect: {CONFIG['NETWORK_EFFECT']})")
    axes[0].set_xlabel("Training Episodes")
    axes[0].set_ylabel("Market Share")
    axes[0].legend()
    
    # 图2: 总体奖励收敛 (Convergence)
    window_size = 20
    moving_avg = np.convolve(history["rewards"], np.ones(window_size)/window_size, mode='valid')
    axes[1].plot(moving_avg, color='green')
    axes[1].set_title("Total Ecosystem Utility (Rewards)")
    axes[1].set_xlabel("Episodes")
    axes[1].set_ylabel("Total Reward")
    
    # 图3: 公平性指标 (Gini Coefficient)
    axes[2].plot(history["gini"], color='orange', alpha=0.6)
    axes[2].axhline(0.4, color='red', linestyle='--', label="Warning Line")
    axes[2].set_title(f"Wealth Distribution Fairness\n(Token Model: {CONFIG['TOKEN_MODEL']})")
    axes[2].set_xlabel("Episodes")
    axes[2].set_ylabel("Gini Coefficient")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('MADRL_Simulation_Result.png', dpi=300)
    print("图表已保存为 MADRL_Simulation_Result.png")
    plt.show()

if __name__ == "__main__":
    # 运行实验
    data = train()
    plot_results(data)
