import sys

path = sys.argv[1]

c, s = 0, 0.

avg_first = 0.
avg_last = 0.
nb_first_classes = 0
col_index = -1

with open(path, 'r') as f:
    for line_index, line in enumerate(f):
        split = line.split(',')
        a = split[-1]
        a = float(a)
        s += a
        c += 1
        step = line.split(',')[0]

        if line_index == 0:
            for col_index in range(1, len(split)):
                if split[col_index] == "x": break
        elif col_index > -1:
            if len(split[1:col_index]) == 0:
                avg_first = 0.
            else:
                avg_first = sum([float(i) for i in split[1:col_index] if i not in ('x', 'X')]) / len(split[1:col_index])
            if len(split[col_index:-1]) == 0:
                avg_last = 0.
            else:
                avg_last = sum([float(i) for i in split[col_index:-1] if i not in ('x', 'X')]) / len(split[col_index:-1])



print(f"Last Step: {step}")
print(f"Final Mean IoU {round(100 * a, 2)}")
print(f'Average Mean IoU {round(100 * s / c, 2)}')
print(f'Mean IoU first {round(100 * avg_first, 2)}')
print(f'Mean IoU last {round(100 * avg_last, 2)}')
