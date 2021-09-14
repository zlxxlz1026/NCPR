# NCPR
### **模型图**

![image-20210914163808950](assets/image-20210914163808950.png)

### **对话推荐系统展示**

* 实时性
* 多样性
* 交互性

#### 客户端与服务器端通信

![时序图](assets/时序图-1630906853918.png)

#### 线程间通信

​	使用经典的多线程并发协作模型——生产者-消费者模型。

![7](assets/7.png)

#### 前后端的双向通信

​	使用WebSocket协议进行前后端的通信。

![6](assets/6.png)

#### 对话推荐系统功能展示

模型选择页面：

![5-1](assets/5-1.png)

用户交互过程：

![5-2](assets/5-2.png)

系统做出用户心仪推荐

![5-3](assets/5-3.png)

选择测试集用户：

![5-4](assets/5-4.png)

显示测试的样例信息：

![5-5](assets/5-5.png)

NCPR经过7轮交互做出成功推荐：

![5-6](assets/5-6.png)

交互达到最大轮次则推荐失败：

![5-7](assets/5-7.png)