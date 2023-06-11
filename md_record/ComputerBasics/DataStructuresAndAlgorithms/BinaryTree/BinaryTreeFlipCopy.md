# 翻转、复制二叉树

## 翻转二叉树：

给定一棵二叉树，将其翻转，即左右子树交换位置。
我们可以使用递归来解决这个问题。对于每个节点，我们交换它的左右子树，并递归的交换它的左右子树。如果当前节点为空，则返回 None。
下面是 Python 语言的实现代码：

    class TreeNode:
        def __init__(self, val=0, left=None, right=None):
            self.val = val
            self.left = left
            self.right = right
    
    class Solution:
        def invertTree(self, root: TreeNode) -&gt; TreeNode:
            if not root:
                return None
        left = self.invertTree(root.left)
        right = self.invertTree(root.right)
    
        root.left, root.right = right, left
    
        return root

### 复制二叉树：

给定一棵二叉树，将其复制。复制后的每个节点包括它的所有子节点。
我们可以使用递归的方法来解决这个问题。对于每个节点，我们分别复制它的左右子树，并用复制后的左右子树建立一个新的节点。如果当前节点为空，则返回 None。
下面是 Python 语言的实现代码：

    class TreeNode:
        def __init__(self, val=0, left=None, right=None):
            self.val = val
            self.left = left
            self.right = right
    
    class Solution:
        def copyTree(self, root: TreeNode) -&gt; TreeNode:
            if not root:
                return None
        node = TreeNode(root.val)
        node.left = self.copyTree(root.left)
        node.right = self.copyTree(root.right)
    
        return node

时间复杂度为 $O(n)$，其中 $n$ 是树中节点的个数。在每个节点上最多只会遍历一次，因此时间复杂度为线性的。同时，由于需要存储复制后的节点，空间复杂度也为 $O(n)$。

- 