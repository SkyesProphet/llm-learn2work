#1.
import sys
def solution(n):
    cnt = 0
    d = 1
    while True:
        pow10 = 10 ** d
        f = pow10 + 1
        lo = 10 ** (d - 1)
        hi = pow10 - 1
        if lo * f > n:
            break
        kmax = n // f
        if kmax >= lo:
            cnt += min(hi, kmax) - lo + 1
        d += 1
    return cnt
def main():
    data = sys.stdin.read().split()
    if not data: return
    n = int(data[0])
    print(solution(n))
if __name__ == '__main__':
    main()



# 2.
import sys

def min_insertion_cost(word1: str, word2: str) -> int:
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # 初始化：空串变成另一个串，需要插入另一个串所有字符
    for i in range(m - 1, -1, -1):
        dp[i][n] = dp[i + 1][n] + ord(word1[i])
    for j in range(n - 1, -1, -1):
        dp[m][j] = dp[m][j + 1] + ord(word2[j])

    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            if word1[i] == word2[j]:
                dp[i][j] = dp[i + 1][j + 1]
            else:
                cost_insert_from_word2 = ord(word2[j]) + dp[i][j + 1]
                cost_insert_from_word1 = ord(word1[i]) + dp[i + 1][j]
                dp[i][j] = min(cost_insert_from_word2, cost_insert_from_word1)

    return dp[0][0]


def main():
    data = sys.stdin.read().strip().splitlines()
    if not data:
        return
    # 按题目输入格式调整；这里默认前两行分别是 word1 和 word2
    word1 = data[0].strip()
    word2 = data[1].strip() if len(data) >= 2 else ""
    print(min_insertion_cost(word1, word2))


if __name__ == "__main__":
    main()
