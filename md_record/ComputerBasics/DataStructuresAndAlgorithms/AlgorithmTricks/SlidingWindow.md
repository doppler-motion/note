**给定一个含有 n 个正整数的数组和一个正整数 target 。找出该数组中满足其和 ≥ target 的长度最小的连续子数组 [numsl, numsl+1, …, numsr-1, numsr] ，并返回其长度。如果不存在符合条件的子数组，返回 0 。**

这一题可以使用滑动窗口的思想来解决。

首先将左指针left和右指针right都指向数组的第一个位置，然后用sum来记录[left,right]范围内所有数的和sums。然后移动右指针，直到sums≥target，接着再移动左指针，直到sums

代码实现如下：

```
def minSubArrayLen(nums, target):
    left, right = 0, 0
    window_sum = nums[0]
    n = len(nums)
    ans = float("inf")

    while right < n:
        if window_sum < target:
            right += 1
            if right == n:
                break
            window_sum += nums[right]
        else:
            ans = min(ans, right - left + 1)
            window_sum -= nums[left]
            left += 1

    if ans == float("inf"):
        return 0
    else:
        return ans

```

首先，我们定义 left 和 right 两个指针，一开始它们都指向数组的第一个位置。然后定义变量 window_sum 表示[left,right]范围内所有数的和，使用变量 ans 表示最短的连续子数组长度。

在 while 循环中，如果 windowsum sum 中；否则，更新 ans 的值并将左指针向右移动，同时从 windowsum 中删除对应元素的值，直到 windowsum < target。

最后，如果 ans 的值没有被更新，说明并没有子数组满足条件，返回 0；否则返回 ans。

时间复杂度为O(n)，空间复杂度为O(1)。



