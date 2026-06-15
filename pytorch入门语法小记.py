# import torch
# x = torch.rand(5,5)
# y = torch.rand(5,5)
# print(x)
# print(y)
# print(x+y)
# print(torch.mul(x,y))

#两者相同
# print(x.size())
# print(x.shape)


 
# import numpy as np
# import torch
 
# # 标量Tensor求导
# # 求 f(x) = a*x**2 + b*x + c 的导数
# x = torch.tensor(-2.0, requires_grad=True)#定义x是自变量，requires_grad=True表示需要求导 
# #也就是说 这里x是自变量 可以变化->可以求导 所以需要requires_grad=True 一直追踪（允许求导）
# a = torch.tensor(1.0)   #这是一个0维张量 也就是只有一个数 1维张量是一维数组
# b = torch.tensor(2.0)
# c = torch.tensor(3.0)
# y = a*torch.pow(x,2)+b*x+c
# y.backward() # backward求得的梯度会存储在自变量x的grad属性中
# dy_dx =x.grad
# print(dy_dx)
# #这时候发现多元函数求导道理与上述一致


 
# import torch
 
# #单个自变量求导
# # 求 f(x) = a*x**4 + b*x + c 的导数
# x = torch.tensor(1.0, requires_grad=True)
# a = torch.tensor(1.0)
# b = torch.tensor(2.0)
# c = torch.tensor(3.0)
# y = a * torch.pow(x, 4) + b * x + c
# #create_graph设置为True,允许创建更高阶级的导数
# #create_graph=True: 保留计算图，这样我们可以对结果继续求导
# #使用 torch.autograd.grad 计算 y 对 x 的一阶导数
# dy_dx = torch.autograd.grad(y, x, create_graph=True)[0]
# # [0]: grad返回的是元组，取第一个元素（因为只有一个输入变量x）
# #求二阶导
# dy2_dx2 = torch.autograd.grad(dy_dx, x, create_graph=True)[0]
# #求三阶导
# dy3_dx3 = torch.autograd.grad(dy2_dx2, x)[0]# 这次不需要 create_graph=True，因为我们不再求更高阶导数了

# print(dy_dx.data, dy2_dx2.data, dy3_dx3)
 
 
# # 多个自变量求偏导
# x1 = torch.tensor(1.0, requires_grad=True)
# x2 = torch.tensor(2.0, requires_grad=True)
# y1 = x1 * x2
# y2 = x1 + x2
# #   只有一个因变量,正常求偏导
# #   计算 y1 对 [x1, x2] 的偏导数
# dy1_dx1, dy1_dx2 = torch.autograd.grad(outputs=y1, inputs=[x1, x2], retain_graph=True)
# #   outputs: 要微分的因变量
# #   inputs: 要对哪个/哪些变量求导
# #   retain_graph=True: 保留计算图，因为后面还要用
# print(dy1_dx1, dy1_dx2)

# # 若有多个因变量，则对于每个因变量,会将求偏导的结果加起来
# dy1_dx, dy2_dx = torch.autograd.grad(outputs=[y1, y2], inputs=[x1, x2])
# # 这里 outputs 是一个列表 [y1, y2]，inputs 是一个列表 [x1, x2]
# # 返回的是 (∂(y1+y2)/∂x1, ∂(y1+y2)/∂x2)  也就是这么写默认y=y1+y2
# # 计算：
# #   ∂(y1+y2)/∂x1 = ∂y1/∂x1 + ∂y2/∂x1 = x2 + 1 = 2 + 1 = 3
# #   ∂(y1+y2)/∂x2 = ∂y1/∂x2 + ∂y2/∂x2 = x1 + 1 = 1 + 1 = 2   
# dy1_dx, dy2_dx
# print(dy1_dx, dy2_dx)
# # 注意：变量名有点误导，这里 dy1_dx 实际上是 ∂(y1+y2)/∂x1
# # dy2_dx 实际上是 ∂(y1+y2)/∂x2
# # 所以 dy1_dx = 3.0, dy2_dx = 2.0



#利用自动微分和优化器求最小值
import numpy as np
import torch
 
# f(x) = a*x**2 + b*x + c的最小值
x = torch.tensor(0.0, requires_grad=True)  # x需要被求导
# 初始化x=0.0作为起始点
# requires_grad=True表示我们要计算f(x)对x的导数
a = torch.tensor(1.0)
b = torch.tensor(-2.0)
c = torch.tensor(1.0)
# 定义二次函数的系数
# 这是已知的，我们不需要优化a,b,c
# 所以没有设置requires_grad=True
optimizer = torch.optim.SGD(params=[x], lr=0.01)  #SGD为随机梯度下降
# 创建优化器，这里用SGD（随机梯度下降）
# params=[x] 告诉优化器：要优化的参数是x
# lr=0.01 是学习率，控制每次更新的大小
# 学习率太小：收敛慢；学习率太大：可能错过最小值
print(optimizer)
 
def f(x):
    result = a * torch.pow(x, 2) + b * x + c
    return (result)
 
for i in range(500):
# 开始500次迭代，每次迭代都调整x，使f(x)更小
    optimizer.zero_grad()  #将模型的参数初始化为0
    # 将模型的参数梯度初始化为0
    # 关键！清除x之前的梯度
    # 因为PyTorch默认会累加梯度，不清零的话梯度会越加越大
    y = f(x)
    y.backward()  #反向传播计算梯度
    optimizer.step()  #更新所有的参数
print("y=", y.data, ";", "x=", x.data)