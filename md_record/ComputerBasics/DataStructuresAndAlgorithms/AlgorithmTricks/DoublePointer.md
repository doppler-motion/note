**给你 n 个非负整数 a1，a2，…，an，每个数代表坐标中的一个点 (i, ai) 。在坐标内画 n 条垂直线，垂直线 i 的两个端点分别为 (i, ai) 和 (i, 0)。找出其中的两条线，使得它们与 x 轴共同构成的容器可以容纳最多的水。**

该问题属于经典的双指针算法问题，其思路如下：

设置左右指针 left 和 right 分别指向数组的起始位置和结束位置；

计算当前指针指向的两个线段的面积，并将其与历史最大面积比较，取较大值；

如果 left 指向的高度 hleft 小于 right 指向的高度 hright，则将 left 右移一位，即 left = left + 1；

如果 hleft 大于等于 hright，则将 right 左移一位，即 right = right - 1。

重复步骤 2-4 直到 left 和 right 已经重合。

代码如下：

```python
def maxArea(height):
    """
    :type height: List[int]
    :rtype: int
    """
    left, right = 0, len(height) - 1
    max_area = 0
    while left < right:
        h_left, h_right = height[left], height[right]
        max_area = max(max_area, min(h_left, h_right) * (right - left))
        if h_left < h_right:
            left += 1
        else:
            right -= 1
    return max_area
```

该算法的时间复杂度为 $O(n)$，空间复杂度为 $O(1)$。

**寻找座位，给定一个列表，0表示未占，1表示已占，寻找与相邻已占位置都最大的空闲位置。至少保证一个已占，一个未占**

这道题可以使用双指针算法进行解决。

具体来说，我们可以使用两个指针 $left$ 和 $right$，初始时，$left$ 指向第一个已占位置的右侧，$right$ 指向第一个未占位置的右侧。然后我们将两个指针向右移动，每次移动时，如果区间 $[left, right]$ 左右两侧都有已占座位，则将 $left$ 向右移动一位，否则，我们就需要计算区间 $[left, right]$ 的长度，将其与之前所有区间长度的最大值进行比较并更新最大值，然后将 $left$ 向右一位，直到左侧出现已占位置，然后继续移动 $right$。当 $right$ 移动到末尾时，我们也需要将区间 $[left, right]$ 的长度和之前所有区间长度的最大值进行比较并更新最大值。

以下是示例代码

```
def find_max_gap(seats):
    left = None  # 最左侧的已占座位位置
    right = 0  # 未占座位位置
    max_gap = 0  # 最大间隔数

    while right < len(seats):
        if seats[right] == 1:
            if left is not None:
                cur_gap = (right - left - 1)
                max_gap = max(max_gap, cur_gap)
            left = right
        
        right += 1
    
    # 检查最右侧是否为占据座位
    if seats[-1] == 0:
        cur_gap = (right - left - 1)
        max_gap = max(max_gap, cur_gap)
    
    return max_gap
```

在这个示例代码中，我们记录了最左侧的已占座位位置、未占座位位置和最大间隔数。我们使用两个指针 $left$ 和 $right$ 对列表进行遍历，当指针 $right$ 指向一个未占位置时，如果 $left$ 指向的位置左侧有已占座位，则计算区间 $[left, right]$ 的长度并更新最大间隔数，然后将 $left$ 向右移动。如果 $left$ 指向的位置左侧没有已占座位，则继续将 $right$ 向右移动。当 $right$ 移动到列表末尾时，我们检查最后一个位置是否为未占位置，如果是的话，我们计算区间 $[left, right]$ 的长度并更新最大间隔数。

需要注意的是，在这个算法中，我们假设了至少有一个已占位置和一个未占位置，如果不满足这个条件，那么这个算法就不适用。