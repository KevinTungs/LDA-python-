# 2152952 自动化 孔维涛
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Hei'] # 设置中文显示字体
plt.rcParams['axes.unicode_minus'] = False   # 正确显示负号

# 数据集
data = np.array([
    [0.666, 0.091, 1],
    [0.243, 0.267, 1],
    [0.244, 0.056, 1],
    [0.342, 0.098, 1],
    [0.638, 0.16, 1],
    [0.656, 0.197, 1],
    [0.359, 0.369, 1],
    [0.592, 0.041, 1],
    [0.718, 0.102, 1],
    [0.697, 0.46, 0],
    [0.774, 0.376, 0],
    [0.633, 0.263, 0],
    [0.607, 0.317, 0],
    [0.555, 0.214, 0],
    [0.402, 0.236, 0],
    [0.481, 0.149, 0],
    [0.436, 0.21, 0],
    [0.557, 0.216, 0]
])

# 分离特征和标签
X = data[:, :2]
y = data[:, 2]

# 计算正例和反例的均值向量
mu_0 = np.mean(X[y == 0], axis=0)
mu_1 = np.mean(X[y == 1], axis=0)

# 计算类内散度矩阵
S_w = np.zeros((2, 2))
for i in range(len(X)):
    if y[i] == 0:
        S_w += np.outer(X[i] - mu_0, X[i] - mu_0)
    else:
        S_w += np.outer(X[i] - mu_1, X[i] - mu_1)

# 计算最优投影方向
w_prime = np.linalg.inv(S_w).dot(mu_0 - mu_1)

# 计算LDA分割线（决策边界），决策边界垂直于投影向量w_prime，且通过两个类别均值的中点
mid_point = (mu_0 + mu_1) / 2
slope = w_prime[1] / w_prime[0]

# 计算垂直于投影向量的斜率
slope_perpendicular = -1 / slope

# 计算截距
intercept_perpendicular = mid_point[1] - slope_perpendicular * mid_point[0]

# 绘制数据点
plt.figure(figsize=(10, 6))
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='正例')
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='反例')

# 绘制LDA分割线
x_values = np.linspace(0, 1, 100)  # x值
y_values = slope_perpendicular * x_values + intercept_perpendicular  # 对应的y值
plt.plot(x_values, y_values, color='green', label='LDA 分割线')

# 其他图形设置
plt.xlabel('属性1')
plt.ylabel('属性2')
plt.legend()
plt.title('数据集和LDA分割线')
plt.show()

# 输出投影向量
print(f'投影向量为：{w_prime}')
# 输出LDA分割线的方程
print(f'LDA分割线的方程为：y = {slope_perpendicular}x + {intercept_perpendicular}')
