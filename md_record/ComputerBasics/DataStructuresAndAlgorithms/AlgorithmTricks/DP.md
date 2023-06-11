**给一个字典，包含几个单词，再给一个string，问字典里的单词能不能拼成string，可以重复使用 Python实现**

可以通过Python实现这样的功能。实现的核心思想是，对于给定字符串，按照长度递增的方式来截取字符串子串，如果一个子串在字典中存在，就从剩余的字符串中截取出新的子串进行处理，直到处理完所有的子串或者无法拼接出目标字符串。

下面是一个示例代码，实现了判断一个字典中的单词能否拼成目标字符串的功能。

```
def can_form_string(word_dict, target_string):
    if not word_dict or not target_string:
        return False
    dp = [False] * (len(target_string) + 1)
    dp[0] = True
    for i in range(len(target_string)):
        if not dp[i]:
            continue
        for word in word_dict:
            if target_string[i:].startswith(word):
                dp[i + len(word)] = True
    return dp[-1]

# 示例调用
word_dict = ['apple', 'pen']
target_string = 'applepenapple'
result = can_form_string(word_dict, target_string)
print(result) # True
```

在示例中，我们首先判断输入的字典和目标字符串是否为空，如果是，则直接返回 False。接着，我们定义一个布尔型数组 dp，其中 dp[i] 表示在目标字符串的前 i 个字符中是否能够被拆分成字典中的单词。初始时，dp[0] 为 True，表示空字符串可以被拆分成字典中的单词。然后我们按照长度递增的方式遍历目标字符串，并在每个位置 i 时判断目标字符串的第 i 个字符是否能够被拆分成字典中的单词。

在内层循环中，我们遍历字典中的每个单词，如果目标字符串第 i 个字符开始的子串能够匹配上字典中的某一个单词，就将 dp[i+len(word)] 标记为 True，表示目标字符串的前 i+len(word) 个字符可拆分成字典中的单词。

最终返回 dp[-1]，表示整个目标字符串是否可以被拆分成字典中的单词。

**给出一个字符串数组，输出这些字符串的最长公共子串。例：[abc, abcde, zzabco]，输出：abc；Python**

求字符串数组的最长公共子串，可以使用动态规划的方法来解决。具体实现方法如下：

定义一个长度为m*n的二维数组matrix，其中m为字符串数组中字符串的最大长度，n为字符串数组的长度。二维数组中第i行第j列的值表示第j个字符串前i个字符组成的子串与其他字符串的最长公共子串长度。

初始化matrix第一行的值为0，第一列的值为0。

从matrix[1][1]开始，遍历二维数组matrix。如果matrix[i][j]表示的字符相同，那么matrix[i][j]的值为matrix[i-1][j-1]的值加1，否则matrix[i][j]的值为0。

遍历所有的matrix[i][j]的值，找出其中的最大值maxV和对应的最大值索引maxI，最长的公共子串为字符串数组中第maxI个字符串的[maxV-len, maxV]子串。

完整代码如下：

```
def longest_common_substring(strs):
    # 获取字符串数组中字符串的最大长度
    n = len(strs)
    max_len = len(max(strs, key=len))

    # 构建二维数组matrix
    matrix = [[0]*n for i in range(max_len)]

    # 初始化matrix第一行和第一列的值
    for i in range(n):
        matrix[0][i] = 0
    for i in range(max_len):
        matrix[i][0] = 0

    # 遍历matrix，更新每个位置的值
    maxV = 0  # 最长公共子串的长度
    maxI = 0  # 最长公共子串所对应的索引
    for i in range(1, max_len):
        for j in range(1, n):
            if strs[j][i-1] == strs[maxI][maxV]:
                matrix[i][j] = matrix[i-1][j-1] + 1
            else:
                matrix[i][j] = 0

            if matrix[i][j] > maxV:
                maxV = matrix[i][j]
                maxI = j
        
    # 返回最长公共子串
    if maxV == 0:
        return ''
    else:
        return strs[maxI][maxV - max_len:maxV]

```

时间复杂度为O(nm^2)，空间复杂度为O(nm)，其中n为字符串数组的长度，m为字符串数组中字符串的最大长度。

