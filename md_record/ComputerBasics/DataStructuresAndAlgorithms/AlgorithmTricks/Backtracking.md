**给出一个数字n，然后给出所有符合括号规律的列表。**

这是一道经典的括号生成问题。我们可以使用回溯算法来解决。

在回溯算法中，我们可以递归地生成所有符合要求的括号字符串。具体实现如下：

```
def generateParenthesis(n):
    def backtrack(cur_str, left, right):
        if len(cur_str) == 2 * n:
            res.append(cur_str)
            return
        if left < n:
            backtrack(cur_str + '(', left + 1, right)
        if right < left:
            backtrack(cur_str + ')', left, right + 1)

    res = []
    backtrack('', 0, 0)
    return res
```

主函数中定义一个回溯函数 backtrack(curstr, left, right)，其中 curstr 表示当前的括号字符串，left 表示已经放置的左括号的数目，right 表示已经放置的右括号的数目。当 left == n 且 right == n 时，说明已经放置了 n 对括号且这个字符串合法，将其加入结果列表 res 中。

接着递归生成括号字符串。如果 left < n，我们可以放置一个左括号；如果 right < left，我们可以放置一个右括号。剩下的情况就是放置右括号比左括号多，此时不满足括号匹配规则，不做操作。

最后调用回溯函数 backtrack('', 0, 0) 来生成所有符合要求的括号字符串。

时间复杂度为O(2^2n)，空间复杂度为O(2n)。



**给你一个字符串，通过在字符串中加入空格使得字符串被分解为几个单词，输出所有结果，**

这是一个常见的字符串分割问题，可以使用回溯算法或动态规划算法来解决。

回溯算法：

回溯算法可以使用递归的方式实现。我们可以从字符串的第一个字符开始，依次枚举每个位置加上空格或者不加空格，判断新的字符串是否仍为有效的单词，如果是，则继续向后递归。同时，我们需要维护一个变量，用来记录当前已经生成的字符串。

下面是回溯算法的示例代码：

```
def word_break(s):
    words = ["apple", "pen", "applepen", "pineapple"]
    res = []

    def backtrack(start, path):
        if start == len(s):
            res.append(path[:-1])
            return
        for i in range(start, len(s)):
            if s[start:i+1] in words:
                backtrack(i+1, path + s[start:i+1] + " ")

    backtrack(0, "")
    return res

```

在代码中，我们首先定义了一个列表 words，用于存储有效的单词。然后，我们定义一个回溯函数 backtrack，该函数接受两个参数，一个表示当前遍历到的位置，另一个表示当前已经生成的字符串。

在回溯函数中，我们首先判断是否已经遍历到了字符串的末尾，如果是，则将当前生成的字符串加入结果列表 res 中，然后返回。否则，我们从当前位置开始依次往后枚举每个位置，并判断新的字符串是否为有效的单词，如果是，则递归调用 backtrack，并将当前生成的字符串加上这个单词和一个空格。

最终，我们调用回溯函数，并返回结果列表 res。



动态规划算法：

动态规划算法可以使用两个数组来实现，一个数组用来记录以当前位置结尾的子字符串是否为一个有效的单词，另一个数组用来记录字符串的划分方式。我们可以从字符串的第一个位置开始，依次枚举所有子字符串，如果子字符串是一个有效的单词，则更新状态数组。同时，我们还需要记录字符串的划分方式，最终根据状态数组来输出所有的划分方式。

下面是动态规划算法的示例代码：

```
def word_break(s):
    words = ["apple", "pen", "applepen", "pineapple"]
    word_set = set(words)
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True
    # 定义 path 数组，用于记录每个位置的字符串划分情况
    path = [[] for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                # 更新 path 数组
                for p in path[j]:
                    path[i].append(p + [s[j:i]])
                if not path[j]:
                    path[i].append([s[j:i]])

    return [' '.join(words) for words in path[n]]
```

在代码中，我们首先定义了一个列表 words，用于存储有效的单词，以及一个集合 word_set，用来优化单词的查找。然后，我们定义了两个变量，一个表示字符串的长度 n，另一个表示当前结尾位置的状态。

接下来，我们定义了两个数组，一个表示状态数组 dp，用来记录以当前位置结尾的子字符串是否为一个有效的单词，另一个表示字符串的划分方式 path。

我们首先将 dp[0] 设置为 True，表示空字符串为一个有效的单词，并初始化 path 数组。然后，从字符串的第一个位置开始，依次枚举所有子字符串，如果子字符串是一个有效的单词，则将 dp[i] 标记为 True，并根据 dp[j] 和 path[j] 更新 dp[i] 和 path[i]。其中，path[i] 存储在字符串 s 的第 i 个位置结尾时的所有划分方式。

最终，我们根据状态数组 dp 和路径数组 path 来输出所有字符串的划分方式。

总的来说，回溯算法适用于字符串较小的情况，而动态规划算法适用于字符串较长的情况，具有更高的效率。使用哪一种算法，应根据具体的情况来选择。





**给定一组元素，元素里有重复情况，在不使用递归的情况下，找到所有不重复的元素全排列**

可以使用回溯算法来实现，在不使用递归的情况下找到所有不重复的元素全排列。回溯算法的基本思路是：从第一个元素开始，依次枚举所有可能的情况，如果当前情况满足条件，则继续递归下去，否则回溯到上一级，重新选择其他可能的情况，直到找到所有符合条件的情况为止。

在本题中，我们可以使用一个哈希表来记录每个元素出现的次数，每次在选择下一个元素时，从哈希表中选择一个没有被选择过的元素，将其加入结果列表中，继续递归下去，直到找到所有符合条件的情况为止。

具体实现过程如下：

对输入的元素进行排序，便于去重；

对每个元素出现的次数进行计数，存储在哈希表中；

对于当前选择的元素，从哈希表中选择一个没有被选择过的元素，并将其加入结果列表中；

如果结果列表中已经包含了所有元素，则将其加入最终结果集中，并回溯到上一级，重新选择其他可能的情况；

如果结果列表中没有包含所有元素，则继续递归下去，直到找到所有符合条件的情况为止。

代码实现如下：

```
def permuteUnique(nums):
    results = [] # 最终结果集
    freq = {} # 哈希表，用于记录每个元素的出现次数
    # 统计每个元素出现的次数
    for num in nums:
        freq[num] = freq.get(num, 0) + 1

    # 回溯函数
    def backtrack(path):
        if len(path) == len(nums):
            results.append(path.copy())
            return
        for num in freq:
            if freq[num] &gt; 0:
                freq[num] -= 1
                path.append(num)
                backtrack(path)
                path.pop()
                freq[num] += 1
    
    # 对元素进行排序，便于去重
    nums.sort()
    # 回溯找所有可能排列
    backtrack([])
    return results
```

该算法的时间复杂度为O(n!)，即所有可能的排列情况。空间上，相较于递归实现方案，由于没有使用递归，所以只需要O(n)的空间来存储结果和哈希表。





**给出一个字符串 s ，拆分该字符串，并返回拆分后唯一子字符串的最大数目。字符串 s 拆分后可以得到若干非空子字符串 ，这些子字符串连接后应当能够还原为原字符串。但是拆分出来的每个子字符串都必须是唯一的 。**

这一题可以使用回溯算法来解决。具体实现方法如下：

定义一个哈希表 used，存储当前已使用的字符串。

定义一个变量 max_count，表示最大的子字符串数目。

以字符串 s 的每个位置 i 为起点，向右截取长度为 1 ~ len(s) - i 的子字符串。如果当前截取的子字符串不在 used 中，则将其加入 used 中，并递归截取后面的子字符串。

如果当前已经遍历完整个字符串 s，更新 max_count 的值。

回溯操作，将当前子字符串从 used 中删除。

完整代码如下：

```
def maxUniqueSplit(s: str) -> int:
    def backtrack(start):
        nonlocal max_count
        if start == len(s):
            max_count = max(max_count, len(used))
            return
        for i in range(start, len(s)):
            cur_str = s[start:i+1]
            if cur_str not in used:
                used.add(cur_str)
                backtrack(i+1)
                used.remove(cur_str)
    used, max_count = set(), 0
    backtrack(0)
    return max_count

```

时间复杂度为O(2^n)，空间复杂度取决于哈希表 used 的大小。



- 八皇后，全排列，组合

八皇后问题是一个经典的计算机科学和数学难题，它的目标是将八个棋子放在一个8×8的棋盘上，使得任意两个棋子都不能在同一行、同一列或同一对角线上。这个问题可以通过搜索算法来解决，其中一个最简单的方法是回溯法。
全排列是指由给定个数的元素所能排列出的所有可能顺序，例如，对于三个不同的元素（1、2、3），可以有六种排列组合：123、132、213、231、312、321。全排列可以通过递归算法实现。
组合是指从给定的元素集合中选取指定数量的元素的所有可能组合。例如，对于三个不同的元素（1、2、3），从其中选择两个元素可以有三种不同的组合：{1，2}、{1，3}、{2，3}。组合也可以通过递归算法来实现，通常使用回溯法和二进制计数法两种方法。



以下是在 Python 中实现八皇后问题、全排列和组合的示例代码：
八皇后问题：

```
def queens(num=8, state=()):
    if len(state) == num - 1:
        for pos in range(num):
            if not conflict(state, pos):
                yield (pos, )
    else:
        for pos in range(num):
            if not conflict(state, pos):
                for result in queens(num, state + (pos,)):
                    yield (pos,) + result

def conflict(state, nextX):
    nextY = len(state)
    for i in range(nextY):
        if abs(state[i] - nextX) in (0, nextY - i):
            return True
    return False

print(list(queens(8)))
```



全排列：

```
def permute(data):
    if len(data) == 0:
        return []
    if len(data) == 1:
        return [data]
    res = []
    for i in range(len(data)):
        rem = data[:i] + data[i+1:]
        for p in permute(rem):
            res.append([data[i]] + p)
    return res

print(permute([1, 2, 3]))
```



组合：

```
def combinations(data, n):
    if n == 0:
        return [[]]
    res = []
    for i in range(len(data)):
        item = data[i]
        remLst = data[i+1:]
        for c in combinations(remLst, n-1):
            res.append([item]+c)
    return res

print(combinations([1, 2, 3, 4], 2))
```



- 重复数字的排列，重复数字的组合

重复数字的排列是指可以包含重复数字的全排列，例如，对于三个数字（1、2、2），可以有五种排列组合：122、212、221、112、121。实现方法可以在全排列基础上略作修改，仍然使用递归、回溯等方法。
重复数字的组合是指可以包含重复数字的组合，例如，对于三个数字（1、2、2），从其中选择两个元素可以有以下四种不同的组合：{1，2}、{1，2}、{2，2}、{2，2}。实现方法可以在组合基础上略作修改，修改枚举元素时需要注意重复的情况。
下面是在 Python 中实现重复数字的排列和组合的示例代码：
重复数字的排列：

```
def permuteUnique(nums):
    def backtrack(idx):
        if idx == n:
            res.append(nums[:])
            return
        used = set()
        for i in range(idx, n):
            if nums[i] not in used:
                used.add(nums[i])
                nums[idx], nums[i] = nums[i], nums[idx]
                backtrack(idx+1)
                nums[idx], nums[i] = nums[i], nums[idx]
    n = len(nums)
    res = []
    backtrack(0)
    return res

print(permuteUnique([1,2,2]))
```



重复数字的组合：

```
def combineWithDuplicates(candidates, target):
    def backtrack(start, target, path):
        if target == 0:
            res.append(path[:])
            return
        for i in range(start, len(candidates)):
            if i &gt; start and candidates[i] == candidates[i-1]:
                continue
            if candidates[i] &gt; target:
                break
            path.append(candidates[i])
            backtrack(i, target-candidates[i], path)
            path.pop()
    candidates.sort()
    res = []
    backtrack(0, target, [])
    return res

print(combineWithDuplicates([1,2,2,2], 4))
```



