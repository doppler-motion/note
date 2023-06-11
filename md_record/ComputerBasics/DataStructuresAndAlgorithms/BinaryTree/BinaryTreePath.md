# 二叉树路径和相关



## 二叉树的路径和为定值的路径

给定一棵二叉树和一个整数sum，寻找所有从根节点到叶子节点路径上的节点值之和等于sum的路径。
我们可以使用递归的方法来解决这个问题。对于每个节点，我们分别计算包括该节点和不包括该节点的路径和，递归地将问题转化为子问题。如果当前节点是叶子节点，并且经过该节点的路径和满足要求，我们将该路径添加到结果列表中。
下面是 Python 语言的实现代码：

    class TreeNode:
        def __init__(self, val=0, left=None, right=None):
            self.val = val
            self.left = left
            self.right = right
    
    class Solution:
        def pathSum(self, root: TreeNode, targetSum: int) -&gt; List[List[int]]:
            self.res = []
        def dfs(node: TreeNode, path: List[int], path_sum: int) -&gt; None:
            if not node:
                return
    
            path.append(node.val)
            path_sum += node.val
    
            if not node.left and not node.right and path_sum == targetSum:
                self.res.append(path[:])
    
            dfs(node.left, path, path_sum)
            dfs(node.right, path, path_sum)
    
            path.pop()
    
        dfs(root, [], 0)
        return self.res

时间复杂度为 $O(n)$，其中 $n$ 是树中节点的个数。在每个节点上最多只会遍历一次，因此时间复杂度为线性的。同时，由于需要存储所有满足要求的路径，空间复杂度也为 $O(n)$。



## **一道二叉树路径和最大**

题目描述

给定一个非空二叉树，返回其最大路径和。

本题中，路径被定义为一条从树中任意节点出发，沿着父节点-子节点连接的边，达到任意节点的序列。该路径至少包含一个节点，且不一定经过根节点。

解法分析

这道题的难点在于如何处理“路径”的定义。由于路,径必须是从一个节点出发，到达另一个节点的一条边上的序列，所以可能存在这样的情况：以某个节点作为路径的起点，往左走到某个节点，再往右走到某个节点，然后返回原来的节点。这样的路径并不能直接计算，但是可以通过计算经过该节点的所有路径，来更新答案。

具体来说，我们可以考虑二叉树中的任意一个节点，计算以该节点作为路径终点的最大路径和。为了确保路径不重复计算，我们只需要计算：从该节点向下走的最大路径和，以及从该节点向左或向右走的最大路径和。这两个路径的和就是以该节点作为路径终点的最大路径和。对于所有节点，我们都计算出以它们作为路径终点的最大路径和，然后取最大值即可。

最后需要注意的是，路径可能是负数，所以需要将初始的路径和初始化为一个比较小的值，例如负无穷。

代码实现

以下是 Python 代码实现：

```
class Solution:
    def maxPathSum(self, root: TreeNode) -> int:
        if not root:
            return 0
        
        self.max_sum = float('-inf')
        
        def dfs(node):
            if not node:
                return 0
            
            # 计算以该节点为路径终点的最大路径和
            left_sum = dfs(node.left)
            right_sum = dfs(node.right)
            cur_sum = node.val + max(0, left_sum) + max(0, right_sum)
            
            # 更新答案
            self.max_sum = max(self.max_sum, cur_sum)
            
            # 返回从该节点向下走的最大路径和
            return node.val + max(0, max(left_sum, right_sum))
        
        dfs(root)
        return self.max_sum
```

时间复杂度为 O(n)，其中 n 是二叉树中的节点数。