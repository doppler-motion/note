**给一个二维数组，只有0和1,0是水，1是陆地，问有多少片岛屿 Python实现**

我们可以使用深度优先搜索（DFS）或广度优先搜索（BFS）遍历这个二维数组，统计岛屿的数量。遍历时，对于每个遇到的陆地格子，我们以它为起点向四周扩散，将相邻的陆地格子全部标记为已访问，直到找不到新的邻居格子为止。整个遍历过程中，我们每次发现一个未曾标记过的陆地格子，就将岛屿数量加 1。

下面是基于 DFS 算法的示例代码：

```
def num_islands(grid):
    if not grid:  # 空二维数组
        return 0

    # 定义 DFS 函数
    def dfs(i, j):
        # 边界条件及访问判断：如果当前位置超出边界或已经访问过，直接返回
        if i < 0 or i >= n_row or j < 0 or j >= n_col or grid[i][j] != "1":
            return
        # 标记为已访问
        grid[i][j] = "0"
        # 继续遍历四个方向
        for d in directions:
            dfs(i + d[0], j + d[1])

    n_row, n_col = len(grid), len(grid[0])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 四个方向：上、下、左、右
    n_islands = 0  # 岛屿数量

    # 遍历整个二维数组
    for i in range(n_row):
        for j in range(n_col):
            if grid[i][j] == "1":
                # 如果当前还未访问过且格子是陆地，从该点开始对整个联通块进行 DFS
                dfs(i, j)
                # 遍历完这个联通块，则发现了一个新的岛屿
                n_islands += 1

    return n_islands

```

在这份代码中，我们定义了一个名为 num_islands 的函数，它接受一个二维的 01 数组作为输入，并返回该数组中岛屿的数量。接下来的 DFS 函数将会各个击破，找到并标记整个联通块。

在主函数中，我们首先检查输入数组是否为空，然后遍历整个数组。对于每个陆地格子，如果它还没被访问过，那么这一次搜索将会覆盖整个岛屿。最终，我们返回搜索过程中标记的岛屿数量。

在这个问题中，DFS 和 BFS 在算法原理和时间复杂度上没什么区别，都是 O(m×n)。不过 DFS 空间消耗会比 BFS 多，因为 BFS 倾向于在处理完一层周围的节点后立马清除队列中的该层节点。



**0代表水，1代表陆地，岛由陆地组成且岛内不允许有湖，让你求岛的周长 Python实现**

这是一个图形遍历问题，可以使用深度优先遍历或广度优先遍历来解决。

深度优先遍历：

深度优先遍历可以使用递归函数实现。我们可以从陆地的任意一个位置开始递归地向四个方向遍历，每次遍历到一个新的位置，就将周长加上与上一个位置的边界数，并标记该位置已经被访问过。同时，我们需要判断遍历的范围是否越界或者该位置为水域，如果是，则不继续遍历。

下面是深度优先遍历（DFS）的示例代码：

```
def islandPerimeter(grid):
    def dfs(i, j):
        nonlocal perimeter
        if not (0 <= i < m and 0 <= j < n) or grid[i][j] == 0:
            perimeter += 1
            return
        if grid[i][j] == -1:
            return
        grid[i][j] = -1
        dfs(i+1, j)
        dfs(i-1, j)
        dfs(i, j+1)
        dfs(i, j-1)
        
    m, n = len(grid), len(grid[0])
    perimeter = 0
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1:
                dfs(i, j)
                return perimeter
    return perimeter

```

在代码中，我们首先定义了一个大小为 m x n 的矩阵 grid，以及一个变量 perimeter，用于表示岛的周长。

在遍历矩阵的每个位置时，如果该位置为陆地，则递归地向四个方向遍历，并将周长加上与上一个位置的边界数。如果遍历的范围越界，或者该位置为水域，则将周长加上 1。为了避免重复遍历，我们还需要标记已经访问过的位置，在代码中，我们通过将访问过的位置的值设置为 -1 来标记。

最终，我们返回周长 perimeter。

广度优先遍历：

广度优先遍历可以使用队列实现。我们可以从陆地的任意一个位置开始逐层向外遍历，每次遍历到一个新的位置，就将周长加上与上一个位置的边界数，并将该位置加入队列中。同时，我们需要判断遍历的范围是否越界或者该位置为水域，如果是，则不将该位置加入队列中。

下面是广度优先遍历的示例代码：

```
def islandPerimeter(grid):
    m, n = len(grid), len(grid[0])
    perimeter = 0
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1:
                queue = [(i, j)]
                grid[i][j] = -1
                while queue:
                    x, y = queue.pop(0)
                    if x == 0 or grid[x-1][y] == 0:
                        perimeter += 1
                    if x == m-1 or grid[x+1][y] == 0:
                        perimeter += 1
                    if y == 0 or grid[x][y-1] == 0:
                        perimeter += 1
                    if y == n-1 or grid[x][y+1] == 0:
                        perimeter += 1
                    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < m and 0 <= ny < n and grid[nx][ny] == 1:
                            queue.append((nx, ny))
                            grid[nx][ny] = -1
                return perimeter
    return perimeter
```

在代码中，我们首先定义了一个大小为 m x n 的矩阵 grid，以及一个变量 perimeter，用于表示岛的周长。

在遍历矩阵的每个位置时，如果该位置为陆地，则以该位置为起点，使用广度优先遍历遍历整个岛，并将周长加上与上一个位置的边界数。如果遍历的范围越界，或者该位置为水域，则将周长加上 1。为了避免重复遍历，我们在遍历过每个位置之后，将其标记为 -1。

最终，我们返回周长 perimeter。

总的来说，深度优先遍历和广度优先遍历都可以解决该问题，使用哪一种算法，应根据具体的情况来选择。如果岛的形状比较复杂，建议使用深度优先遍历算法，因为它的回溯特性可以很好地处理岛的凹凸形状。如果岛的形状比较简单，建议使用广度优先遍历算法，因为它的迭代特性比较容易实现。





- 图的搜索

图的搜索是指在图中寻找一条从起点到目标节点的路径的过程。根据搜索的目标和方式，常见的图的搜索算法可以分为以下几种。

深度优先搜索（DFS）：深度优先搜索是一种先将当前节点的所有邻居节点都搜索一遍，直到找到目标节点或者所有路径都搜索完毕，然后回溯到上一个节点继续搜索的方法。使用栈来实现深度优先搜索。
广度优先搜索（BFS）：广度优先搜索是一种先将当前节点的所有邻居节点加入队列，然后逐个出队并继续搜索的方法。使用队列来实现广度优先搜索。
Dijkstra 算法：Dijkstra 算法是一种基于贪心策略的最短路径搜索算法，它通过维护一个到起点距离最短的节点集合和一个距离数组来计算从起点到其他节点的最短路径。使用堆来实现 Dijkstra 算法可以提高效率。
A* 算法：A* 算法是一种启发式搜索算法，它在搜索过程中使用估价函数来评估剩余路径的长度，从而尽可能快地到达目标节点。A* 算法综合了贪心策略和一些高级算法，可以在实际应用中取得良好的效果。

下面是在 Python 中实现深度优先搜索和广度优先搜索的示例代码：
深度优先搜索：

```
def dfs(graph, start):
    visited, stack = set(), [start]
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            stack.extend(graph[vertex] - visited)
    return visited

graph = {'A': set(['B', 'C']),
         'B': set(['A', 'D', 'E']),
         'C': set(['A', 'F']),
         'D': set(['B']),
         'E': set(['B', 'F']),
         'F': set(['C', 'E'])}

print(dfs(graph, 'A'))
```



广度优先搜索：

```
from collections import deque

def bfs(graph, start):
    visited, queue = set(), deque([start])
    while queue:
        vertex = queue.popleft()
        if vertex not in visited:
            visited.add(vertex)
            queue.extend(graph[vertex] - visited)
    return visited

graph = {'A': set(['B', 'C']),
         'B': set(['A', 'D', 'E']),
         'C': set(['A', 'F']),
         'D': set(['B']),
         'E': set(['B', 'F']),
         'F': set(['C', 'E'])}

print(bfs(graph, 'A'))
```

