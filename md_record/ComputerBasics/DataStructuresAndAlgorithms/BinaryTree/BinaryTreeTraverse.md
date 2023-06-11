# 二叉树的遍历

前序遍历：对于当前节点，先输出该节点的值，然后遍历左子树，最后遍历右子树。

中序遍历：对于当前节点，先遍历左子树，然后输出该节点的值，最后遍历右子树。

后序遍历：对于当前节点，先遍历左子树，然后遍历右子树，最后输出该节点的值。

层次遍历：从二叉树的根节点开始，按照从上到下、从左到右的顺序遍历所有节点。

其中，前序遍历、中序遍历和后序遍历可以通过递归或栈的方式实现，而层次遍历通常需要用到队列。

## 前中后遍历的非递归实现

二叉树的前序遍历，中序遍历和后序遍历都可以使用非递归的方式进行。下面分别介绍三种遍历方式的非递归实现：

前序遍历的非递归实现：

前序遍历的非递归实现主要依靠栈来实现。从根节点开始，将根节点压入栈中，然后循环执行以下步骤：

取出栈顶元素，输出该元素的值；
如果该元素的右子树不为空，则将右子树压入栈中；
如果该元素的左子树不为空，则将左子树压入栈中；

重复以上步骤，直到栈为空，所有节点已被遍历。
代码如下：

```
def preorderTraversal(root):
    if not root:
        return []
    stack, res = [root], []
    while stack:
        node = stack.pop()
        res.append(node.val)
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    return res
```




中序遍历的非递归实现：

中序遍历的非递归实现也依靠栈来实现，但是实现起来相对复杂一些。具体而言，需要先将所有左子节点压入栈中，然后从栈中取出一个节点，输出该节点的值，并将该节点的右子树压入栈中，以待之后遍历。重复以上步骤，直到栈为空，所有节点已被遍历。
代码如下：

```
def inorderTraversal(root):
    if not root:
        return []
    stack, res = [], []
    node = root
    while node or stack:
        while node:
            stack.append(node)
            node = node.left
        node = stack.pop()
        res.append(node.val)
        node = node.right
    return res
```




后序遍历的非递归实现：

后序遍历的非递归实现相对前两个要更加复杂一些，需要使用到两个栈。具体而言，需要将节点按照根-右-左的顺序压入第一个栈中，然后再将第一个栈中的节点全部弹出，按照左-右-根的顺序压入第二个栈中。最后输出第二个栈中的所有节点即可。
代码如下：

```
def postorderTraversal(root):
    if not root:
        return []
    stack1, stack2, res = [root], [], []
    while stack1:
        node = stack1.pop()
        stack2.append(node)
        if node.left:
            stack1.append(node.left)
        if node.right:
            stack1.append(node.right)
    while stack2:
        res.append(stack2.pop().val)
    return res
```

总的来说，这三种二叉树遍历的非递归实现方法都可以通过借助栈来实现。在具体实现时，需要根据不同的遍历方式选择不同的压栈和弹栈顺序，从而实现对应的遍历效果。

- 层次遍历，之字遍历？

二叉树层次遍历是按照从上到下、从左到右的顺序遍历二叉树的所有节点。通常可以使用队列来实现层次遍历，从根节点开始将节点压入队列中，然后依次遍历队列中的节点，将每个节点的左右子节点压入队列中。在遍历完一层节点后，遍历下一层节点。具体的过程如下：

```
def levelOrder(root):
    if not root:
        return []
    queue, res = [root], []
    while queue:
        level_result = []
        level_size = len(queue)
        for i in range(level_size):
            node = queue.pop(0)
            level_result.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        res.append(level_result)
    return res
```

如果要进行之字遍历，可以在上述代码的基础上稍作修改。具体而言，需要在遍历每一层节点时，隔一层将遍历结果翻转即可。具体的过程如下：

```
def zigzagLevelOrder(root):
    if not root:
        return []
    queue, res = [root], []
    while queue:
        level_result = []
        level_size = len(queue)
        for i in range(level_size):
            node = queue.pop(0)
            level_result.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        if len(res) % 2 == 1:
            level_result.reverse()
        res.append(level_result)
    return res
```

例如，对于下面的二叉树：
    3
   / \
  9  20
    /  \
   15   7

二叉树层次遍历的结果为：
[[3], [9, 20], [15, 7]]

而二叉树之字形遍历的结果为：
[[3], [20, 9], [15, 7]]

## 二叉树的递归实现

```
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# 前序遍历
def preorderTraversal(root):
    res = []
    def dfs(node):
        if not node:
            return
        res.append(node.val)
        dfs(node.left)
        dfs(node.right)
    dfs(root)
    return res

# 中序遍历
def inorderTraversal(root):
    res = []
    def dfs(node):
        if not node:
            return
        dfs(node.left)
        res.append(node.val)
        dfs(node.right)
    dfs(root)
    return res

# 后序遍历
def postorderTraversal(root):
    res = []
    def dfs(node):
        if not node:
            return
        dfs(node.left)
        dfs(node.right)
        res.append(node.val)
    dfs(root)
    return res

# 层次遍历
def levelOrder(root):
    if not root:
        return []
    res = []
    queue = collections.deque()
    queue.append(root)
    while queue:
        level = []
        size = len(queue)
        for i in range(size):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        res.append(level)
    return res
```



## 二叉树的序列化与反序列化

二叉树的序列化是将二叉树按照某种方式转换为字符串的过程，可以用于在网络上传输二叉树，或者将二叉树存储到文件中等。二叉树的反序列化则是将序列化后的字符串转换回原来的二叉树的过程。以下我们介绍两种序列化与反序列化二叉树的常见方式。
前序遍历序列化与反序列化
前序遍历二叉树的顺序是先遍历根节点，然后遍历左子树，最后遍历右子树。因此，我们可以通过前序遍历将一个二叉树转换为字符串。
在序列化二叉树时，对于每个节点，我们先将节点的值转换为字符串并拼接到字符串序列化结果中，然后遍历左子树和右子树，递归进行序列化。如果节点为 None，则将 "null" 作为节点的值拼接到字符串序列化结果中。
下面是前序遍历序列化二叉树的Python代码实现：

```
def serialize(root):
    if not root:
        return 'null'
    return str(root.val) + ',' + serialize(root.left) + ',' + serialize(root.right)
```



在反序列化二叉树时，我们需要先将序列化字符串按照逗号分隔符拆分成字符串数组，然后依次取出每个节点的值创建对应的节点，同时递归构建左子树和右子树。如果当前的节点值为 "null"，则返回 None。
下面是前序遍历反序列化二叉树的Python代码实现：

```
def deserialize(data):
    queue = data.split(',')
    def build():
        val = queue.pop(0)
        if val == 'null':
            return None
        node = TreeNode(int(val))
        node.left = build()
        node.right = build()
        return node
    return build()
```



## 层次遍历序列化与反序列化

层次遍历二叉树是按照从上到下依次遍历每一层的节点，对于每一层的节点，先遍历左子树，再遍历右子树。因此，我们可以通过层次遍历将一个二叉树转换为字符串。
在序列化二叉树时，对于每个节点，我们先将节点的值转换为字符串并拼接到字符串序列化结果中，然后将左子树和右子树依次加入到队列中，再从队列中取出下一个节点进行这个过程。如果节点为 None，则将 "null" 作为节点的值拼接到字符串序列化结果中。
下面是层次遍历序列化二叉树的Python代码实现：

```
def serialize(root):
    if not root:
        return 'null'
    queue = [root]
    res = []
    while queue:
        node = queue.pop(0)
        if not node:
            res.append('null')
            continue
        res.append(str(node.val))
        queue.append(node.left)
        queue.append(node.right)
    return ','.join(res)
```



在反序列化二叉树时，我们需要先将序列化字符串按照逗号分隔符拆分成字符串数组，然后首先取出第一个节点的值创建对应的节点作为根节点，将其加入到队列中。然后从队列中取出下一个节点，创建对应的节点作为左子节点，将其加入到队列中，然后从队列中取出下一个节点，创建对应的节点作为右子节点，将其加入到队列中。依次类推，直到队列为空。
下面是层次遍历反序列化二叉树的Python代码实现：

```
def deserialize(data):
    if data == 'null':
        return None
    queue = data.split(',')
    root = TreeNode(int(queue.pop(0)))
    nodes = [root]
    while queue:
        node = nodes.pop(0)
        val = queue.pop(0)
        if val != 'null':
            node.left = TreeNode(int(val))
            nodes.append(node.left)
        if queue:
            val = queue.pop(0)
            if val != 'null':
                node.right = TreeNode(int(val))
                nodes.append(node.right)
    return root
```



使用以上两种方式进行序列化与反序列化操作时，需要注意以下几点：

序列化结果不唯一，同一个二叉树可能会有多种不同的序列化结果。
序列化和反序列化是对应的过程，序列化方式和反序列化方式必须相同，否则无法正确还原二叉树。

## 前中，后中遍历结果恢复二叉树

以下是使用 Python 语言实现的代码：

```
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def buildTree(preorder, inorder):
    if not preorder or not inorder:
        return None
```



    # 建立哈希表，存储中遍历结果中每个元素对应的位置
    inorder_map = {}
    for idx, val in enumerate(inorder):
        inorder_map[val] = idx
    
    root_val = preorder[0]
    root = TreeNode(root_val)
    root_idx = inorder_map[root_val]
    
    root.left = buildTree(preorder[1:1+root_idx], inorder[:root_idx])
    root.right = buildTree(preorder[1+root_idx:], inorder[root_idx+1:])
    
    return root

该函数接受两个列表作为输入，分别为前中遍历结果和后中遍历结果。首先进行边界条件的判断，如果前遍历结果或中遍历结果为空，则直接返回空节点 None。接着递归处理子树，首先找到根节点，即前遍历结果的第一个元素，然后在中遍历结果中找到该元素的位置。根据该位置，可以计算出左子树中元素的个数，然后递归处理左子树和右子树。该函数最后返回根节点 root。
在实现中，我们使用了哈希表 inorder_map 来存储中遍历结果中每个元素对应的位置。在递归过程中，可以直接查询该哈希表，避免了在中遍历结果中搜索对应元素的过程，提高了效率。



## **已知中序和后序，求二叉树**

后序遍历序列的最后一个元素一定是该二叉树的根节点；

在中序遍历序列中，根节点的左边元素一定是该二叉树的左子树中的节点，右边元素一定是该二叉树的右子树中的节点；

根据上一步得到的左右子树的中序遍历序列，我们可以递归地求出左右子树的后序遍历序列，并通过递归重建左右子树。

```
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def buildTree(inorder, postorder):
    """
    :type inorder: List[int]
    :type postorder: List[int]
    :rtype: TreeNode
    """
    def buildTreeHelper(left, right):
        if left > right:
            return None
        val = postorder.pop()
        root = TreeNode(val)
        root_index = inorder.index(val, left, right+1)
        root.right = buildTreeHelper(root_index+1, right)
        root.left = buildTreeHelper(left, root_index-1)
        return root

    return buildTreeHelper(0, len(inorder)-1)
```



该算法的时间复杂度为 $O(n)$，其中 n 为二叉树的节点数量。这是因为对于二叉树的每一个节点，都需要通过中序遍历序列和后序遍历序列求出其左右子树。由于中序遍历序列和后序遍历序列的长度均为 n，因此总时间复杂度为 $O(n^2)$。但是由于我们使用了哈希表来储存中序遍历序列中每个元素的下标，因此我们可以将时间复杂度优化到 $O(n)$。空间复杂度为 $O(n)$，因为我们需要额外使用哈希表和递归栈来存储中序遍历序列和递归调用的上下文。



## 排序二叉树的序列化

```
以下是使用 Python 语言实现的代码：
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def serialize(root):
    if not root:
        return ''

    res = []
    stack = [(0, root)]

    while stack:
        idx, node = stack.pop()
        if node:
            # 将节点的值转换为字符串，加入结果列表
            res.append(str(node.val))
            stack.append((idx + 1, node.right))
            stack.append((idx, node.left))
        else:
            # 如果节点为空，加入占位符 '_'，方便后面反序列化
            res.append('_')

    # 用 '#' 连接结果列表，作为最终的序列化字符串
    return '#'.join(res)

def deserialize(data):
    if not data:
        return None

    # 将序列化字符串按 '#' 分割，得到每个节点的值或占位符
    values = data.split('#')

    def dfs(idx):
        if idx &gt;= len(values) or values[idx] == '_':
            return None

        # 将节点的值转换为整数，创建节点
        node = TreeNode(int(values[idx]))
        node.left = dfs(idx + 1)
        node.right = dfs(idx + 2)

        return node

    # 从根节点开始递归构建二叉树
    return dfs(0)

在序列化函数 serialize 中，我们使用了迭代的方法进行前序遍历。具体来说，我们使用一个栈 stack 来存储每个节点和它的位置。每次从栈中取出一个节点，将节点的值加入到结果列表 res 中。如果节点不为空，将其右子节点和左子节点分别入栈，注意要将右子节点放在左子节点前面，以保证前序遍历的顺序正确。如果节点为空，将占位符 '_' 加入到结果列表中。
在反序列化函数 deserialize 中，我们使用了递归的方法从根节点开始构建二叉树。具体来说，在 dfs 函数中，我们首先判断当前位置 idx 是否越界或者对应节点的值是否为占位符 '_'，如果是，则返回空节点 None。否则，我们创建一个新的节点，将节点的值赋为整数类型的 values[idx]。然后分别递归构建左子树和右子树，将左右子树的根节点分别设为当前节点的左子节点和右子节点。最后返回根节点即可。
注意，在序列化函数中，我们用符号 '_' 作为占位符，因此节点的值中不能出现该字符。在实现中，我们将节点的值转换为字符串类型，因此可以保证字符串中只包含数字字符，避免了这个问题。
```

