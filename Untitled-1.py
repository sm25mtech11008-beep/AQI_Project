nums = [12, 45, 7, 89, 23, 56]
for x in range(len(nums) - 1):
    if nums[x] > nums[x + 1]:
        print(nums[x], "is greater than", nums[x + 1])
