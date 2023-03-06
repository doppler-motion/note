# 刷题记录--剑指offer 专项突击版

[TOC]



## 剑指offer II 010. 和为k的子数组

### 日期 2022-12-07  -- 前缀和

> 给定一个整数数组和一个整数 `k` **，**请找到该数组中和为 `k` 的连续子数组的个数。

* 示例1

```
输入:nums = [1,1,1], k = 2
输出: 2
解释: 此题 [1,1] 与 [1,1] 为两种不同的情况
```

* 示例2

```
输入:nums = [1,2,3], k = 3
输出: 2
```

### 题解

> 前缀和 

```python
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        total = counts = 0
        dic = {}

        for num in nums:
            total += num
            if total == k:
                counts += 1
            if total - k in dic:
                counts += dic[total - k]
            dic[total] = dic.get(total, 0) + 1

        return counts
```

## 剑指offer II 011. 0 和 1 个数相同的子数组

### 日期 2022-12-05  -- 前缀和+哈希表

> 给定一个二进制数组 `nums` , 找到含有相同数量的 `0` 和 `1` 的最长连续子数组，并返回该子数组的长度。

* 示例1

```
输入: nums = [0,1]
输出: 2
说明: [0, 1] 是具有相同数量 0 和 1 的最长连续子数组。
```

* 示例2

```
输入: nums = [0,1,0]
输出: 2
说明: [0, 1] (或 [1, 0]) 是具有相同数量 0 和 1 的最长连续子数组。
```

### 题解

> 前缀和 + 哈希表

```python
class Solution:
    def secondHighest(self, s: str) -> int:
        dct = {0: -1}
        cur = ans = 0
        n = len(nums)
        for i in range(n):
            cur += 2 * nums[i] - 1
            if cur in dct:
                if i - dct[cur] > ans:
                    ans = i - dct[cur]

            else:
                dct[cur] = i

        return ans
```

## 剑指offer II 012. 左右两边子数组的和相等

### 日期 2022-12-05  -- 前缀和

> 给你一个整数数组 nums ，请计算数组的 中心下标 。
>
> 数组 中心下标 是数组的一个下标，其左侧所有元素相加的和等于右侧所有元素相加的和。
>
> 如果中心下标位于数组最左端，那么左侧数之和视为 0 ，因为在下标的左侧不存在元素。这一点对于中心下标位于数组最右端同样适用。
>
> 如果数组有多个中心下标，应该返回 最靠近左边 的那一个。如果数组不存在中心下标，返回 -1 。
>

* 示例1

```
输入：nums = [1,7,3,6,5,6]
输出：3
解释：
中心下标是 3 。
左侧数之和 sum = nums[0] + nums[1] + nums[2] = 1 + 7 + 3 = 11 ，
右侧数之和 sum = nums[4] + nums[5] = 5 + 6 = 11 ，二者相等。
```

* 示例2

```
输入：nums = [1, 2, 3]
输出：-1
解释：
数组中不存在满足此条件的中心下标。
```

* 示例3

```python
输入：nums = [2, 1, -1]
输出：0
解释：
中心下标是 0 。
左侧数之和 sum = 0 ，（下标 0 左侧不存在元素），
右侧数之和 sum = nums[1] + nums[2] = 1 + -1 = 0 。
```

### 题解

> 前缀和
>

```python
class Solution:
    def pivotIndex(self, nums: List[int]) -> int:
        n = len(nums)
        total = sum(nums)
        sum_left = 0
        for i in range(n):
            if (2*sum_left + nums[i]) == total:
                return i
            else:
                sum_left += nums[i]

        return -1
```

