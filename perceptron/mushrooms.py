mushrooms = open("agaricus-lepiota.data", "r")
output = open("mushrooms.csv", "w")

lines = mushrooms.readlines()
count = 0
for line in lines:
    params = line.strip().split(',')
    sample = params[1:]
    nums = list(map(lambda s: ord(s), sample))
    poisoned = 0
    if params[0] == 'p':
        poisoned = 1
    nums.append(poisoned)
    nums.insert(0, count)
    count += 1
    l = len(nums)
    for i in range(l):
        if i < l - 1:
            output.write(f'{nums[i]},')
        else:
            output.write(str(nums[i]))
    output.write('\n')
mushrooms.close()

