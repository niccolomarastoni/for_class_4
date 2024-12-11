import simple_calc as sc

if __name__ == '__main__':
    op = int(input('what operation do you need?\n0 - sum, 1 - sub, 2 - mult'))
    a = int(input('a = '))
    b = int(input('b = '))
    if op == 0:
        print(f'{a} + {b} = {sc.sum(a, b)}')

    else:
        print('not available yet')