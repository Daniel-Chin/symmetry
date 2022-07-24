FILES = [
    'model3 (1d trans, 2d repeat).txt', 
    'model5 (1d free, 2d repeat).txt', 
]

def main():
    results = ([], [])
    for filename, result in zip(FILES, results):
        with open(filename, 'r') as f:
            # reached_end = False
            for line_i, line in enumerate(f):
                if line_i % 4 == 1:
                    try:
                        _, x = line.split('mean_diff: ')
                    except ValueError:
                        break
                    x = float(x)
                if line_i % 4 == 2:
                    _, std = line.split('std: ')
                    std = float(std)
                    result.append(x / std)
    print(*results[0], sep='\n')
    print()
    print(*results[1], sep='\n')

main()
