> 阅读本文默认读者有一定Python基础

[TOC]



### Python函数参数的默认值

#### 问题引入

实现这样一个类方法Example，其中方法中含有变量type, data。在\_\_init\_\_方法中，初始化变量，其中data是一个字典并带有初始值。创建两个实例ex1和ex2, 对ex1中的data赋值。

```python
class Example:
    def __init__(self, type, data={}):
        self.type = type

        self.data = data


# 创建两个实例
ex1 = Example("1")
ex2 = Example("2")
# 为ex1中的data赋值
ex1.data["a"] = 1

# 输出ex1 和 ex2 中的data
print("the value of data in ex1 :", ex1.data)
print("the value of data in ex2 :", ex2.data)
```

各位期待的输出是什么？是这样？

```python
the value of data in ex1 : {'a': 1}
the value of data in ex2 : {}
```

然而，确是这样。

```python
the value of data in ex1 : {'a': 1}
the value of data in ex2 : {'a': 1}
```

我们修改ex1中data的值，连带着ex2中data的值也修改了，这是为何？

#### 原因分析

首先我们看下两个实例中data的地址：

```python
id1 = id(ex1.data)
id2 = id(ex2.data)
print("the address of data in ex1: ", id1)
print("the address of data in ex2: ", id2)

# 输出
the address of data in ex1:  140297792731072
the address of data in ex2:  140297792731072
```

这样原因就很明显了。

___在类初始化时，如果在函数声明中，使用了默认值，那么在生成多个实例时，该变量会使用同一个内存地址进行初始化。___

那么，当一个实例修改了该内存地址的值，则所有类的实例相应的变量值都会改变。

#### 解决方案

那么解决方案是什么？

___尽量避免为参数使用默认值___

修改以上代码如下：

* 示例1

```python
class Example:
    def __init__(self, type):
        self.type = type

        self.data = {}

# 创建两个实例
ex1 = Example("1")
ex2 = Example("2")
# 为ex1中的data赋值
ex1.data["a"] = 1

# 输出ex1 和 ex2 中的data
print("the value of data in ex1 :", ex1.data)
print("the value of data in ex2 :", ex2.data)

# 输出
# the value of data in ex1 : {'a': 1}
# the value of data in ex2 : {}

id1 = id(ex1.data)
id2 = id(ex2.data)
print("the address of data in ex1: ", id1)
print("the address of data in ex2: ", id2)

# 输出
# the address of data in ex1:  140190955419904
# the address of data in ex2:  140190417302144
```

此时输出就正确了，而且内存地址也是不同的。

#### 其他

以上示例，说明在初始赋值为``可变参数``时，如list，dict会出现这种问题；

而默认参数如果是``不可变参数``时，会发生什么情况呢？

* 示例2

```python
class Example:
    def __init__(self, type, cur=1):
        self.type = type

        self.cur = cur

# 创建两个实例
ex1 = Example("1")
ex2 = Example("2")
# 修改ex1中的ext值
ex1.cur = 2

# 输出ex1 和 ex2 中的cur
print("the value of cur in ex1 :", ex1.cur)
print("the value of cur in ex2 :", ex2.cur)

# 输出
# the value of cur in ex1 : 2
# the value of cur in ex2 : 1

id1 = id(ex1.cur)
id2 = id(ex2.cur)
print("the address of cur in ex1: ", id1)
print("the address of cur in ex2: ", id2)

# 输出
# the address of cur in ex1:  4307659136
# the address of cur in ex2:  4307659104
```

从结果上说明，如果是``不可变参数``，__变量不共享__。

#### 总结

在定义函数时，应尽量为参数避免使用__默认值__, 尤其是 ___可变参数___。

最后为大家分享一个案例：

<https://mp.weixin.qq.com/s/d9fI1hTfX5IrXAjRI_n4tg>

