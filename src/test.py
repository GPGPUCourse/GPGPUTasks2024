arr = [3, 5, 12, 22, 45, 64, 69, 82, 17, 29, 35, 73, 86, 90, 95, 99]
block_size = 8
inds = [0 for i in range(2 * block_size)]
left = 0
right = 8

for i in range(1, 2 * block_size + 1):
    lj, rj = 0, i - 1
    if i > block_size:
        rj = 2 * block_size - i - 1
    while (lj <= rj):
        j = int((lj + rj) / 2)
        if arr[left + j + max(0, i - block_size)] < arr[right - j + min(i, block_size) - 1]:
            lj = j + 1
        else:
            rj = j - 1
    inds[i - 1] = lj + max(0, i - block_size)

print(inds)

arr_sorted = [0 for i in range(2 * block_size)]
arr_sorted[0] = min(arr[left], arr[right])
for i in range(1, 2 * block_size):
    if inds[i] > inds[i - 1]:
        arr_sorted[i] = arr[left + max(0, i - block_size + 1) + (inds[i] - max(0, i - block_size + 1)) - 1]
    else:
        arr_sorted[i] = arr[right + min(i, block_size - 1) - (inds[i] - max(0, i - block_size + 1))]
        
print(arr_sorted)